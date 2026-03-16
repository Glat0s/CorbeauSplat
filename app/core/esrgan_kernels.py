"""
Integration layer: inject/remove CorbeauSplat Triton kernels into a loaded
RRDBNet (RealESRGAN generator) model.

Usage
-----
    from app.core.esrgan_kernels import inject_esrgan_kernels, validate_esrgan_kernels

    # Assuming model is an RRDBNet-like architecture
    inject_esrgan_kernels(model)

After injection the model uses:
  - MemEfficientDenseBlock  (pre-allocated concat buffer, no cat() malloc)
  - triton_leakyrelu_inplace (vectorised in-place activation)
  - triton_scale_add         (fused x*0.2 + skip)
  - triton_pixel_shuffle_2x  (fused pixel-shuffle without staging tensor)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("esrgan_kernels")

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_esrgan_kernels(
    model: nn.Module,
    replace_dense_blocks: bool = True,
    replace_pixel_shuffle: bool = True,
    replace_rrdb_residual: bool = True,
) -> int:
    """
    Walk an RRDBNet model and replace supported modules/operations with
    Triton-backed equivalents.  Returns the number of replacements made.
    """
    replaced = 0

    if replace_dense_blocks:
        replaced += _replace_dense_blocks(model)

    if replace_rrdb_residual:
        replaced += _patch_rrdb_residual(model)

    if replace_pixel_shuffle:
        replaced += _replace_pixel_shuffle(model)

    logger.info(
        "ESRGAN Triton kernel injection complete: %d replacement(s). " "triton=%s",
        replaced,
        _TRITON_AVAILABLE,
    )
    return replaced


def remove_esrgan_kernels(model: nn.Module) -> None:
    """
    Remove CorbeauSplat Triton kernel patches from model.
    """
    from app.core.vendor.esrgan_triton_kernels import MemEfficientDenseBlock

    for name, module in list(model.named_modules()):
        if isinstance(module, MemEfficientDenseBlock):
            logger.warning(
                "MemEfficientDenseBlock at '%s' cannot be reverted automatically "
                "(weights are preserved).",
                name,
            )
        if getattr(module, "_triton_rrdb_patched", False):
            module.forward = module._orig_forward
            del module._orig_forward
            del module._triton_rrdb_patched


def validate_esrgan_kernels(
    model: nn.Module,
    tile_size: int = 64,
    device: str = "cuda",
    atol: float = 1e-2,
) -> bool:
    """
    A/B test: run injected vs original model on a small synthetic tile.
    Returns True when outputs agree within *atol*.
    """
    import copy

    ref_model = copy.deepcopy(model).to(device).eval()
    inj_model = copy.deepcopy(model).to(device)
    inject_esrgan_kernels(inj_model)
    inj_model.eval()

    dummy = torch.randn(1, 3, tile_size, tile_size, device=device, dtype=torch.float16)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            ref_out = ref_model(dummy)
            inj_out = inj_model(dummy)

    max_diff = (ref_out.float() - inj_out.float()).abs().max().item()
    passed = max_diff < atol
    logger.info(
        "ESRGAN kernel validation: max_diff=%.4f, atol=%.4f → %s",
        max_diff,
        atol,
        "PASS" if passed else "FAIL",
    )
    return passed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _replace_dense_blocks(model: nn.Module) -> int:
    """Replace ResidualDenseBlock instances with MemEfficientDenseBlock using duck-typing."""
    from app.core.vendor.esrgan_triton_kernels import MemEfficientDenseBlock

    replaced = 0
    for parent_module in model.modules():
        for attr_name, child in list(parent_module.named_children()):
            # Detect RDB by checking for conv1-5 attributes
            if not isinstance(child, MemEfficientDenseBlock) and all(
                hasattr(child, f"conv{i}") for i in range(1, 6)
            ):

                try:
                    new_block = MemEfficientDenseBlock.from_module(child)
                    new_block = new_block.to(
                        device=next(child.parameters()).device,
                        dtype=next(child.parameters()).dtype,
                    )
                    setattr(parent_module, attr_name, new_block)
                    replaced += 1
                except Exception as e:
                    logger.debug("Failed to replace dense block at %s: %s", attr_name, e)

    return replaced


def _patch_rrdb_residual(model: nn.Module) -> int:
    """
    Patch RRDB.forward to use triton_scale_add for the final skip connection.
    """
    from app.core.vendor.esrgan_triton_kernels import triton_scale_add

    patched = 0
    for module in model.modules():
        # Detect RRDB by checking for rdb1, rdb2, rdb3 attributes
        if not (hasattr(module, "rdb1") and hasattr(module, "rdb2") and hasattr(module, "rdb3")):
            continue
        if getattr(module, "_triton_rrdb_patched", False):
            continue

        orig_fwd = module.forward

        def _patched_fwd(x, m=module, _sa=triton_scale_add):
            # Run the three RDB sub-blocks
            out = m.rdb1(x)
            out = m.rdb2(out)
            out = m.rdb3(out)
            return _sa(out, x, scale=0.2)

        module._orig_forward = orig_fwd
        module.forward = _patched_fwd
        module._triton_rrdb_patched = True
        patched += 1

    return patched


def _replace_pixel_shuffle(model: nn.Module) -> int:
    """
    Replace nn.PixelShuffle(2) modules with a TritonPixelShuffle2x wrapper.
    """
    replaced = 0
    for parent_module in model.modules():
        for attr_name, child in list(parent_module.named_children()):
            if isinstance(child, nn.PixelShuffle) and child.upscale_factor == 2:
                setattr(parent_module, attr_name, _TritonPixelShuffle2x())
                replaced += 1
    return replaced


class _TritonPixelShuffle2x(nn.Module):
    """Wraps triton_pixel_shuffle_2x as an nn.Module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from app.core.vendor.esrgan_triton_kernels import triton_pixel_shuffle_2x

        return triton_pixel_shuffle_2x(x)
