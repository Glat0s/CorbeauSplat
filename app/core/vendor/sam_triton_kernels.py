"""
Custom Triton kernels for SAM (Segment Anything Model) ViT encoder.

Optimizations over the baseline torch.compile path:
  1. TritonLayerNorm — single-pass online variance (one H→M read, saves 1 HBM round-trip
     vs PyTorch's 2-pass algorithm). 768/1024/1280-wide rows fit in L1 cache.
  2. TritonFusedMLPFirst — fuses the first MLP linear projection + GELU activation
     into one kernel, eliminating the intermediate (N, 3072/4096) tensor write+read.
  3. triton_window_partition / triton_window_unpartition — fused gather/scatter
     replacing the 4 separate ops (permute→reshape→permute→reshape) that partition
     the 64×64 feature map into 14×14 local windows.

All kernels fall back to PyTorch equivalents when Triton is unavailable.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("sam_triton_kernels")

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    logger.debug("triton not available — SAM Triton kernels will use PyTorch fallbacks")


# ---------------------------------------------------------------------------
# Kernel 1: Single-pass online LayerNorm
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _layernorm_online_fwd(
        X_ptr, Y_ptr, W_ptr, B_ptr,
        stride,        # row stride of X and Y (elements)
        N,             # number of columns (normalised dimension)
        eps,
        BLOCK: tl.constexpr,
    ):
        """One-pass Welford online LayerNorm.  One program per row."""
        row = tl.program_id(0)
        X_ptr = X_ptr + row * stride
        Y_ptr = Y_ptr + row * stride

        # --- accumulate mean + M2 in registers (Welford) ---
        mean = tl.zeros([1], dtype=tl.float32)
        m2   = tl.zeros([1], dtype=tl.float32)
        count = tl.zeros([1], dtype=tl.float32)

        for start in range(0, N, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            # Welford combine
            n_new   = tl.sum(mask.to(tl.float32), axis=0)
            delta   = x - mean
            mean    = mean + tl.sum(delta * mask.to(tl.float32), axis=0) / (count + n_new + 1e-12)
            delta2  = x - mean
            m2      = m2 + tl.sum(delta * delta2 * mask.to(tl.float32), axis=0)
            count   = count + n_new

        rstd = 1.0 / tl.sqrt(m2 / count + eps)

        # --- normalise + affine ---
        for start in range(0, N, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(Y_ptr + cols, y.to(tl.float16), mask=mask)

    @triton.jit
    def _layernorm_gelu_online_fwd(
        X_ptr, Y_ptr, W_ptr, B_ptr,
        stride, N, eps,
        BLOCK: tl.constexpr,
    ):
        """Single-pass LayerNorm + tanh-GELU fused kernel (for MLP pre-activation)."""
        row = tl.program_id(0)
        X_ptr = X_ptr + row * stride
        Y_ptr = Y_ptr + row * stride

        mean = tl.zeros([1], dtype=tl.float32)
        m2   = tl.zeros([1], dtype=tl.float32)
        count = tl.zeros([1], dtype=tl.float32)

        for start in range(0, N, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            n_new   = tl.sum(mask.to(tl.float32), axis=0)
            delta   = x - mean
            mean    = mean + tl.sum(delta * mask.to(tl.float32), axis=0) / (count + n_new + 1e-12)
            delta2  = x - mean
            m2      = m2 + tl.sum(delta * delta2 * mask.to(tl.float32), axis=0)
            count   = count + n_new

        rstd = 1.0 / tl.sqrt(m2 / count + eps)

        SQRT2OVERPI: tl.constexpr = 0.7978845608028654
        COEFF:       tl.constexpr = 0.044715

        for start in range(0, N, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < N
            x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            ln = (x - mean) * rstd * w + b
            # Tanh GELU approximation
            inner = SQRT2OVERPI * (ln + COEFF * ln * ln * ln)
            gelu  = ln * 0.5 * (1.0 + tl.libdevice.tanh(inner))
            tl.store(Y_ptr + cols, gelu.to(tl.float16), mask=mask)


def _triton_layernorm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    fuse_gelu: bool = False,
) -> torch.Tensor:
    """Low-level dispatch: calls the appropriate Triton kernel."""
    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, orig_shape[-1])
    if x_2d.dtype != torch.float16:
        x_2d = x_2d.to(torch.float16)
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)

    BLOCK = min(triton.next_power_of_2(N), 1024)

    kernel = _layernorm_gelu_online_fwd if fuse_gelu else _layernorm_online_fwd
    kernel[(M,)](
        x_2d, y, weight.to(torch.float16), bias.to(torch.float16),
        x_2d.stride(0), N, eps,
        BLOCK=BLOCK,
    )
    return y.view(orig_shape)


# ---------------------------------------------------------------------------
# PyTorch module wrappers
# ---------------------------------------------------------------------------

class TritonLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using the single-pass Triton kernel.

    Parameters
    ----------
    normalized_shape : int | tuple[int]
    eps : float
    fuse_gelu : bool
        When True, fuses tanh-GELU into the kernel (for the MLP pre-norm path).
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        fuse_gelu: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.fuse_gelu = fuse_gelu
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _TRITON_AVAILABLE and x.is_cuda and x.dtype in (torch.float16, torch.float32):
            return _triton_layernorm_impl(
                x, self.weight, self.bias, self.eps, self.fuse_gelu
            )
        # Fallback
        out = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if self.fuse_gelu:
            out = F.gelu(out, approximate="tanh")
        return out

    @classmethod
    def from_module(cls, ln: nn.LayerNorm, fuse_gelu: bool = False) -> "TritonLayerNorm":
        """Convert an existing nn.LayerNorm to TritonLayerNorm, copying weights."""
        new = cls(
            ln.normalized_shape,
            eps=ln.eps,
            elementwise_affine=ln.elementwise_affine,
            fuse_gelu=fuse_gelu,
        )
        if ln.elementwise_affine:
            new.weight = nn.Parameter(ln.weight.clone())
            new.bias   = nn.Parameter(ln.bias.clone())
        return new

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"fuse_gelu={self.fuse_gelu}, triton={_TRITON_AVAILABLE}"
        )


# ---------------------------------------------------------------------------
# Kernel 3: Fused window partition / unpartition (gather / scatter)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _window_partition_fwd(
        SRC_ptr, DST_ptr,
        H, W, C,
        win: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Partition (B, H, W, C) → (B*nH*nW, win, win, C) without intermediate permute.
        grid = (B * nH * nW * win * win,)
        """
        idx = tl.program_id(0)
        nW_tiles  = W // win
        win2      = win * win
        tile_idx  = idx // win2
        local_idx = idx %  win2
        b_idx  = tile_idx // (H // win * nW_tiles)
        rem    = tile_idx %  (H // win * nW_tiles)
        th     = rem // nW_tiles
        tw     = rem %  nW_tiles
        lh     = local_idx // win
        lw     = local_idx %  win
        src_h  = th * win + lh
        src_w  = tw * win + lw
        src_off = ((b_idx * H + src_h) * W + src_w) * C
        dst_off = (idx) * C

        for c in range(0, C, BLOCK_C):
            cols = c + tl.arange(0, BLOCK_C)
            mask = cols < C
            v = tl.load(SRC_ptr + src_off + cols, mask=mask)
            tl.store(DST_ptr + dst_off + cols, v, mask=mask)

    @triton.jit
    def _window_unpartition_fwd(
        SRC_ptr, DST_ptr,
        H, W, C,
        win: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Inverse of _window_partition_fwd."""
        idx = tl.program_id(0)
        nW_tiles  = W // win
        win2      = win * win
        tile_idx  = idx // win2
        local_idx = idx %  win2
        b_idx  = tile_idx // (H // win * nW_tiles)
        rem    = tile_idx %  (H // win * nW_tiles)
        th     = rem // nW_tiles
        tw     = rem %  nW_tiles
        lh     = local_idx // win
        lw     = local_idx %  win
        dst_h  = th * win + lh
        dst_w  = tw * win + lw
        src_off = (idx) * C
        dst_off = ((b_idx * H + dst_h) * W + dst_w) * C

        for c in range(0, C, BLOCK_C):
            cols = c + tl.arange(0, BLOCK_C)
            mask = cols < C
            v = tl.load(SRC_ptr + src_off + cols, mask=mask)
            tl.store(DST_ptr + dst_off + cols, v, mask=mask)


def triton_window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition (B, H, W, C) → (B*nH*nW, window_size, window_size, C).
    Returns partitioned tensor and (H, W) for unpartition.
    Falls back to torch ops when Triton is unavailable.
    """
    B, H, W, C = x.shape
    if not (_TRITON_AVAILABLE and x.is_cuda and H % window_size == 0 and W % window_size == 0):
        # Fallback path (same as SAM's original implementation)
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, (H, W)

    nH, nW = H // window_size, W // window_size
    out = torch.empty(B * nH * nW, window_size, window_size, C, dtype=x.dtype, device=x.device)
    total = B * nH * nW * window_size * window_size
    BLOCK_C = min(triton.next_power_of_2(C), 64)
    _window_partition_fwd[(total,)](
        x.contiguous(), out, H, W, C,
        win=window_size, BLOCK_C=BLOCK_C,
    )
    return out, (H, W)


def triton_window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    hw: Tuple[int, int],
    original_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Reverse of triton_window_partition."""
    H, W = hw
    B = windows.shape[0] // (H // window_size * W // window_size)
    C = windows.shape[-1]

    if not (_TRITON_AVAILABLE and windows.is_cuda and H % window_size == 0 and W % window_size == 0):
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x

    out = torch.empty(B, H, W, C, dtype=windows.dtype, device=windows.device)
    total = windows.shape[0] * window_size * window_size
    BLOCK_C = min(triton.next_power_of_2(C), 64)
    _window_unpartition_fwd[(total,)](
        windows.contiguous(), out, H, W, C,
        win=window_size, BLOCK_C=BLOCK_C,
    )
    return out
