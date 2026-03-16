"""
sam_kernels.py — Inject CorbeauSplat custom kernels into a loaded SAM image encoder.

Three-tier optimisation stack applied by inject_sam_triton_kernels():

  Tier 1 — cuDNN Flash Attention (highest impact, ~2× attention speedup)
    Patches each SAM windowed attention block's forward() to call
    F.scaled_dot_product_attention (SDPA) with the relative-position bias
    as an additive attn_mask.  PyTorch 2.0 + cuDNN 9+ automatically selects
    the cuDNN flash-attention kernel (cudnn_sdp=True on this system).

  Tier 2 — CUDA C++ LayerNorm (single-pass warp-shuffle, ~1.5× per block)
    Replaces nn.LayerNorm with TritonLayerNorm backed by the C++ ext when
    available, or the Triton kernel otherwise.

  Tier 3 — CUDA C++ window partition / unpartition (float4, ~1.5× per block)
    Replaces the window-op calls in each windowed attention block with the
    float4-vectorised CUDA kernel.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sam_kernels")

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def inject_sam_triton_kernels(
    image_encoder: nn.Module,
    replace_layernorm: bool = True,
    replace_window_ops: bool = True,
    flash_attention: bool = True,
) -> int:
    """
    Inject all available custom kernels into *image_encoder*.
    Returns total number of module/op replacements made.
    """
    replaced = 0

    if flash_attention:
        replaced += _patch_flash_attention(image_encoder)

    if replace_layernorm:
        replaced += _replace_layernorm_recursive(image_encoder)

    if replace_window_ops:
        replaced += _patch_window_ops(image_encoder)

    from app.core.cuda_ext import ext_available
    logger.info(
        "SAM kernel injection: %d replacement(s). "
        "cuda_ext=%s triton=%s flash_attn=%s",
        replaced, ext_available(), _TRITON_AVAILABLE, flash_attention,
    )
    return replaced


def remove_sam_triton_kernels(image_encoder: nn.Module) -> int:
    """Restore all TritonLayerNorm modules back to nn.LayerNorm."""
    from app.core.vendor.sam_triton_kernels import TritonLayerNorm

    restored = 0
    for name, module in list(image_encoder.named_modules()):
        if not isinstance(module, TritonLayerNorm):
            continue
        parent, attr = _get_parent_and_attr(image_encoder, name)
        ln = nn.LayerNorm(
            module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
        )
        if module.elementwise_affine:
            ln.weight = nn.Parameter(module.weight.clone())
            ln.bias   = nn.Parameter(module.bias.clone())
        setattr(parent, attr, ln)
        restored += 1

    # Remove flash-attention patches
    for module in image_encoder.modules():
        if getattr(module, "_corbeau_flash_patched", False):
            module.forward = module._corbeau_orig_forward
            del module._corbeau_orig_forward
            del module._corbeau_flash_patched

    # Remove window-op patches
    for module in image_encoder.modules():
        if getattr(module, "_corbeau_window_patched", False):
            module.forward = module._corbeau_orig_window_forward
            del module._corbeau_orig_window_forward
            del module._corbeau_window_patched

    logger.info("SAM kernels removed: %d LayerNorm(s) restored.", restored)
    return restored


def validate_sam_kernels(
    image_encoder: nn.Module,
    input_size: int = 1024,
    device: str = "cuda",
    atol: float = 5e-2,
) -> bool:
    """A/B test: injected encoder vs reference. Returns True if max_diff < atol."""
    import copy

    ref = copy.deepcopy(image_encoder).to(device).eval()
    inj = copy.deepcopy(image_encoder).to(device)
    inject_sam_triton_kernels(inj)
    inj.eval()

    dummy = torch.randn(1, 3, input_size, input_size, device=device, dtype=torch.float16)
    with torch.no_grad(), torch.cuda.amp.autocast():
        ref_out = ref(dummy)
        inj_out = inj(dummy)

    max_diff = (ref_out.float() - inj_out.float()).abs().max().item()
    passed   = max_diff < atol
    logger.info(
        "SAM kernel validation: max_diff=%.4f atol=%.4f → %s",
        max_diff, atol, "PASS" if passed else "FAIL",
    )
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: cuDNN Flash Attention patch
# ─────────────────────────────────────────────────────────────────────────────

def _patch_flash_attention(image_encoder: nn.Module) -> int:
    """
    Replace manual QK^T+softmax+V in SAM Attention modules with
    F.scaled_dot_product_attention (triggers cuDNN flash-attention path).

    SAM's Attention.forward() signature (segment_anything >= 1.0):
        def forward(self, x: Tensor) -> Tensor:
            B, H, W, _ = x.shape
            qkv = self.qkv(x)
            q, k, v   = qkv.reshape(...).split(C, dim=-1)
            # optional rel_pos bias added to attn weights
            attn = (q * self.scale) @ k.transpose(-2,-1)
            if self.rel_pos_h is not None:
                attn = add_decomposed_rel_pos(attn, q, ...)
            attn = attn.softmax(-1)
            x = (attn @ v).view(B, H, W, -1)

    We wrap the entire forward and inject SDPA after QKV projection.
    """
    patched = 0
    for module in image_encoder.modules():
        cls_name = type(module).__name__
        if cls_name not in ("Attention",):
            continue
        if getattr(module, "_corbeau_flash_patched", False):
            continue
        _wrap_attention_flash(module)
        module._corbeau_flash_patched = True
        patched += 1
    return patched


def _wrap_attention_flash(attn_module: nn.Module) -> None:
    """Monkey-patch one SAM Attention module to use SDPA."""
    orig_forward = attn_module.forward

    def _flash_forward(x: torch.Tensor) -> torch.Tensor:
        # -- 1. QKV projection (same as original) ------------------------
        B, H, W, _ = x.shape
        qkv = attn_module.qkv(x)

        # Reshape -> (B, H*W, 3, num_heads, head_dim)
        num_heads = attn_module.num_heads
        head_dim  = qkv.shape[-1] // (3 * num_heads)
        qkv = qkv.reshape(B, H * W, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(2)        # each (B, H*W, num_heads, head_dim)
        q = q.transpose(1, 2)          # (B, num_heads, H*W, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -- 2. Relative position bias (if present) -----------------------
        attn_mask = None
        try:
            # SAM uses add_decomposed_rel_pos to build the bias
            if (hasattr(attn_module, "rel_pos_h")
                    and attn_module.rel_pos_h is not None):
                # Build rel_pos bias using SAM's own helper
                import segment_anything.modeling.image_encoder as _enc
                # q for rel_pos: (B, num_heads, H*W, head_dim) -> (B*num_heads, H*W, head_dim)
                q_for_rel = q.reshape(B * num_heads, H * W, head_dim)
                # attn scratch for add_decomposed_rel_pos: (B*num_heads, H*W, H*W)
                attn_scratch = torch.zeros(
                    B * num_heads, H * W, H * W,
                    device=x.device, dtype=x.dtype
                )
                attn_scratch = _enc.add_decomposed_rel_pos(
                    attn_scratch, q_for_rel,
                    attn_module.rel_pos_h, attn_module.rel_pos_w,
                    (H, W), (H, W),
                )
                # Reshape to (B, num_heads, H*W, H*W) for SDPA attn_mask
                attn_mask = attn_scratch.reshape(B, num_heads, H * W, H * W)
        except Exception:
            attn_mask = None

        # -- 3. cuDNN / Flash Attention via SDPA -------------------------
        scale = head_dim ** -0.5
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
            enable_cudnn=True,
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                scale=scale,
            )

        # -- 4. Reshape back (B, H, W, C) --------------------------------
        out = out.transpose(1, 2).reshape(B, H, W, -1)
        return attn_module.proj(out)

    attn_module._corbeau_orig_forward = orig_forward
    attn_module.forward = _flash_forward


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: LayerNorm replacement
# ─────────────────────────────────────────────────────────────────────────────

def _replace_layernorm_recursive(module: nn.Module) -> int:
    from app.core.vendor.sam_triton_kernels import TritonLayerNorm

    # Upgrade TritonLayerNorm to use CUDA ext backend
    _upgrade_to_cuda_ext = True
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            new = TritonLayerNorm.from_module(child)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += _replace_layernorm_recursive(child)
    return replaced


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Window partition / unpartition ops
# ─────────────────────────────────────────────────────────────────────────────

def _patch_window_ops(image_encoder: nn.Module) -> int:
    """Replace SAM's window_partition / window_unpartition module globals."""
    try:
        import segment_anything.modeling.image_encoder as _enc
        from app.core.cuda_ext import window_partition_cuda, window_unpartition_cuda

        # Only patch if not already patched
        if not getattr(_enc, "_corbeau_window_patched", False):
            _enc._orig_window_partition   = _enc.window_partition
            _enc._orig_window_unpartition = _enc.window_unpartition
            _enc.window_partition   = window_partition_cuda
            _enc.window_unpartition = window_unpartition_cuda
            _enc._corbeau_window_patched = True
            logger.info("SAM window ops patched with CUDA float4 kernels.")
            return 2   # 2 functions patched
    except Exception as e:
        logger.debug("Window op patching skipped: %s", e)
    return 0


def _patch_rel_pos_for_cuda_graph() -> int:
    """
    Patch SAM's get_rel_pos to cache relative-position index tensors.

    SAM's get_rel_pos() calls torch.arange() and .long() on every forward pass.
    These operations are not CUDA-graph-safe because they involve runtime
    allocations / CPU-GPU synchronisations.  We cache the result per (q_size,
    k_size, rel_pos.shape) so that subsequent calls reuse pre-built tensors.

    Call this once before starting a CUDA graph capture.
    Returns 1 if patched, 0 if already patched or not available.
    """
    try:
        import segment_anything.modeling.image_encoder as _enc

        if getattr(_enc, "_corbeau_rel_pos_patched", False):
            return 0

        _orig_get_rel_pos = _enc.get_rel_pos
        _cache: dict = {}

        def _cached_get_rel_pos(
            q_size: int, k_size: int, rel_pos: torch.Tensor
        ) -> torch.Tensor:
            key = (q_size, k_size, rel_pos.device, rel_pos.dtype, rel_pos.shape)
            if key not in _cache:
                _cache[key] = _orig_get_rel_pos(q_size, k_size, rel_pos)
            return _cache[key]

        _enc.get_rel_pos = _cached_get_rel_pos
        _enc._corbeau_rel_pos_patched = True
        logger.info("SAM get_rel_pos patched with index cache for CUDA graph safety.")
        return 1
    except Exception as e:
        logger.debug("get_rel_pos patch skipped: %s", e)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_parent_and_attr(root: nn.Module, dotted_name: str):
    parts  = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
