"""
Integration layer: inject/remove CorbeauSplat Triton kernels into a loaded
SAM image encoder.

Usage
-----
    from app.core.sam_kernels import inject_sam_triton_kernels, validate_sam_kernels

    sam_model = sam_model_registry[model_type](checkpoint=ckpt)
    inject_sam_triton_kernels(sam_model.image_encoder)

After injection the image encoder uses:
  - TritonLayerNorm   (single-pass online variance) for all LayerNorm modules
  - Triton window partition / unpartition for windowed attention blocks (when
    the block has window_size > 0 and the input is divisible by window_size)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("sam_kernels")

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inject_sam_triton_kernels(
    image_encoder: nn.Module,
    replace_layernorm: bool = True,
    replace_window_ops: bool = True,
) -> int:
    """
    Walk the SAM image encoder and replace supported modules with Triton versions.

    Returns the number of modules replaced.
    """
    if not _TRITON_AVAILABLE:
        logger.warning("Triton not available — SAM kernel injection skipped.")
        return 0

    from app.core.vendor.sam_triton_kernels import TritonLayerNorm

    replaced = 0

    # --- Replace nn.LayerNorm → TritonLayerNorm ---
    if replace_layernorm:
        replaced += _replace_layernorm_recursive(image_encoder)

    # --- Patch window attention blocks ---
    if replace_window_ops:
        replaced += _patch_window_attention(image_encoder)

    logger.info(
        "SAM Triton kernel injection complete: %d module(s) replaced. "
        "triton=%s", replaced, _TRITON_AVAILABLE
    )
    return replaced


def remove_sam_triton_kernels(image_encoder: nn.Module) -> int:
    """
    Restore all TritonLayerNorm modules back to nn.LayerNorm.
    Returns the number of modules restored.
    """
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

    logger.info("SAM Triton kernels removed: %d module(s) restored.", restored)
    return restored


def validate_sam_kernels(
    image_encoder: nn.Module,
    input_size: int = 1024,
    device: str = "cuda",
    atol: float = 5e-2,
) -> bool:
    """
    Run a quick A/B comparison: injected encoder vs reference PyTorch encoder.

    Returns True when the outputs agree within *atol* (FP16 tolerance).
    """
    import copy

    ref_encoder = copy.deepcopy(image_encoder).to(device)
    tri_encoder = copy.deepcopy(image_encoder).to(device)
    inject_sam_triton_kernels(tri_encoder)

    ref_encoder.eval()
    tri_encoder.eval()

    dummy = torch.randn(1, 3, input_size, input_size, device=device, dtype=torch.float16)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            ref_out = ref_encoder(dummy)
            tri_out = tri_encoder(dummy)

    max_diff = (ref_out.float() - tri_out.float()).abs().max().item()
    passed   = max_diff < atol
    logger.info(
        "SAM kernel validation: max_diff=%.4f, atol=%.4f → %s",
        max_diff, atol, "PASS" if passed else "FAIL"
    )
    return passed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replace_layernorm_recursive(module: nn.Module) -> int:
    """DFS: replace every nn.LayerNorm child with TritonLayerNorm."""
    from app.core.vendor.sam_triton_kernels import TritonLayerNorm

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, TritonLayerNorm.from_module(child))
            replaced += 1
        else:
            replaced += _replace_layernorm_recursive(child)
    return replaced


def _patch_window_attention(image_encoder: nn.Module) -> int:
    """
    Monkey-patch the forward method of SAM's windowed attention blocks to use
    Triton window partition / unpartition kernels.

    SAM block forward signatures vary between versions.  We detect the window
    partitioning call by inspecting `block.window_size` and wrapping the
    `forward` method with a patched version that calls our Triton ops.
    """
    from app.core.vendor.sam_triton_kernels import (
        triton_window_partition,
        triton_window_unpartition,
    )

    patched = 0
    for module in image_encoder.modules():
        ws = getattr(module, "window_size", 0)
        if ws <= 0:
            continue
        if getattr(module, "_triton_window_patched", False):
            continue

        _patch_block_window(module, ws)
        module._triton_window_patched = True
        patched += 1

    return patched


def _patch_block_window(block: nn.Module, window_size: int) -> None:
    """Replace the standard window ops in one SAM block with Triton variants."""
    from app.core.vendor.sam_triton_kernels import (
        triton_window_partition,
        triton_window_unpartition,
    )

    original_forward = block.forward

    def _patched_forward(x: torch.Tensor) -> torch.Tensor:
        # Shortcut: delegate everything to the original forward but intercept
        # the SAM-internal window_partition / window_unpartition calls.
        # We achieve this by temporarily replacing the global references that
        # SAM's forward function resolves at call-time from the module.
        import segment_anything.modeling.image_encoder as _enc_mod
        _orig_wp  = getattr(_enc_mod, "window_partition",   None)
        _orig_wup = getattr(_enc_mod, "window_unpartition", None)

        if _orig_wp is not None:
            _enc_mod.window_partition   = triton_window_partition
        if _orig_wup is not None:
            _enc_mod.window_unpartition = triton_window_unpartition

        try:
            out = original_forward(x)
        finally:
            if _orig_wp  is not None:
                _enc_mod.window_partition   = _orig_wp
            if _orig_wup is not None:
                _enc_mod.window_unpartition = _orig_wup
        return out

    block.forward = _patched_forward


def _get_parent_and_attr(root: nn.Module, dotted_name: str):
    """Return (parent_module, attribute_name) for a dotted path like 'blocks.3.norm1'."""
    parts  = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
