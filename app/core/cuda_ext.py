"""
cuda_ext.py — JIT-compile and cache the CorbeauSplat CUDA extension.

Provides Python wrappers for all kernels in csrc/corbeau_kernels.cu:
  - layer_norm_fwd(x, weight, bias, eps, fuse_gelu) → Tensor
  - window_partition_fwd(x, window_size) → Tensor
  - window_unpartition_fwd(windows, window_size, H, W) → Tensor
  - leaky_relu_scale_add(x, residual, neg_slope, scale) → Tensor
  - pixel_shuffle_2x(x) → Tensor

Falls back gracefully to Triton / PyTorch equivalents when the extension
cannot be compiled (no MSVC, missing CUDA, etc.).

The extension is compiled once and cached in the torch extension cache dir.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger("cuda_ext")

_EXT = None          # loaded extension module, or None
_EXT_TRIED = False   # True after first load attempt (avoids repeated retries)

CSRC_DIR = Path(__file__).parent / "csrc"

# ─────────────────────────────────────────────────────────────────────────────
# Build environment setup (Windows / MSVC / CUDA 12.4)
# ─────────────────────────────────────────────────────────────────────────────

_VS_INSTALL = r"D:\Microsoft Visual Studio\2022\Community"
_MSVC_VER   = "14.37.32822"
_CUDA_12_4  = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
_WINSDK_VER = "10.0.22621.0"
_WINSDK_ROOT = r"D:\Windows Kits\10"


def _setup_msvc_env() -> dict:
    """Return os.environ copy with MSVC + Windows SDK variables set."""
    env = os.environ.copy()
    msvc_bin = rf"{_VS_INSTALL}\VC\Tools\MSVC\{_MSVC_VER}\bin\Hostx64\x64"
    sdk_bin  = rf"{_WINSDK_ROOT}\bin\{_WINSDK_VER}\x64"
    sdk_inc  = rf"{_WINSDK_ROOT}\Include\{_WINSDK_VER}"
    sdk_lib  = rf"{_WINSDK_ROOT}\Lib\{_WINSDK_VER}"
    msvc_inc = rf"{_VS_INSTALL}\VC\Tools\MSVC\{_MSVC_VER}\include"
    msvc_lib = rf"{_VS_INSTALL}\VC\Tools\MSVC\{_MSVC_VER}\lib\x64"

    env["PATH"] = msvc_bin + ";" + sdk_bin + ";" + env.get("PATH", "")
    env["INCLUDE"] = (
        msvc_inc + ";"
        + rf"{sdk_inc}\ucrt" + ";"
        + rf"{sdk_inc}\um" + ";"
        + rf"{sdk_inc}\shared" + ";"
        + env.get("INCLUDE", "")
    )
    env["LIB"] = (
        msvc_lib + ";"
        + rf"{sdk_lib}\ucrt\x64" + ";"
        + rf"{sdk_lib}\um\x64" + ";"
        + env.get("LIB", "")
    )
    env["LIBPATH"] = msvc_lib + ";" + env.get("LIBPATH", "")
    # Force CUDA 12.4 to match torch
    env["CUDA_HOME"]  = _CUDA_12_4
    env["CUDA_PATH"]  = _CUDA_12_4
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    return env


def _load_ext() -> Optional[object]:
    """JIT-compile corbeau_ext and return the module, or None on failure."""
    global _EXT, _EXT_TRIED
    if _EXT_TRIED:
        return _EXT
    _EXT_TRIED = True

    if not torch.cuda.is_available():
        logger.info("CUDA not available — CUDA extension skipped.")
        return None

    src_cu  = CSRC_DIR / "corbeau_kernels.cu"
    src_cpp = CSRC_DIR / "corbeau_ext.cpp"
    if not src_cu.exists() or not src_cpp.exists():
        logger.warning("CUDA extension sources not found in %s", CSRC_DIR)
        return None

    try:
        from torch.utils.cpp_extension import load

        env = _setup_msvc_env()
        # Temporarily inject env vars so that torch.utils.cpp_extension can
        # find cl.exe and the correct nvcc
        original_env = {}
        for k, v in env.items():
            original_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            ext = load(
                name="corbeau_cuda",
                sources=[str(src_cpp), str(src_cu)],
                extra_cuda_cflags=[
                    "-arch=sm_89",       # RTX 4090 (Ada Lovelace)
                    "--use_fast_math",
                    "-O3",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
                extra_cflags=["/O2", "/std:c++17"],
                verbose=False,
            )
            logger.info("CorbeauSplat CUDA extension loaded (sm_89, CUDA 12.4).")
            _EXT = ext
        finally:
            # Restore original environment
            for k, v in original_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    except Exception as e:
        logger.warning("CUDA extension build failed (%s) — using Triton/PyTorch fallbacks.", e)
        _EXT = None

    return _EXT


# ─────────────────────────────────────────────────────────────────────────────
# Public Python wrappers (auto-select CUDA ext → Triton → PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def layer_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
    fuse_gelu: bool = False,
) -> torch.Tensor:
    """
    Single-pass warp-shuffle LayerNorm (RTX 4090 optimised).
    Falls back to Triton → PyTorch when unavailable.
    """
    ext = _load_ext()
    if ext is not None and x.is_cuda and x.dtype == torch.float16:
        orig = x.shape
        x2 = x.contiguous().view(-1, orig[-1])
        y  = ext.layer_norm_fwd(x2, weight.half(), bias.half(), eps, fuse_gelu)
        return y.view(orig)

    # Triton fallback
    try:
        from app.core.vendor.sam_triton_kernels import _triton_layernorm_impl
        return _triton_layernorm_impl(x, weight, bias, eps, fuse_gelu)
    except Exception:
        pass

    # PyTorch fallback
    out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    if fuse_gelu:
        out = F.gelu(out, approximate="tanh")
    return out


def window_partition_cuda(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    (B, H, W, C) → (B*nH*nW, win, win, C).
    Matches SAM's window_partition API: pads x to next multiple of window_size
    and returns (windows, (padded_H, padded_W)).
    """
    B, H, W, C = x.shape

    # Pad to next multiple of window_size (mirrors SAM's original implementation)
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    ext = _load_ext()
    if (ext is not None and x.is_cuda and x.dtype == torch.float16 and C % 8 == 0):
        return ext.window_partition_fwd(x.contiguous(), window_size), (Hp, Wp)

    # Triton fallback
    try:
        from app.core.vendor.sam_triton_kernels import triton_window_partition
        windows, _ = triton_window_partition(x, window_size)
        return windows, (Hp, Wp)
    except Exception:
        pass

    # PyTorch fallback
    x2 = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x2.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C), (Hp, Wp)


def window_unpartition_cuda(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Reverse of window_partition_cuda.
    Matches SAM's window_unpartition(windows, window_size, pad_hw, hw) API.
    pad_hw = (padded_H, padded_W); hw = (orig_H, orig_W) for crop-back.
    """
    Hp, Wp = pad_hw
    C = windows.shape[-1]
    B = windows.shape[0] // (Hp // window_size * Wp // window_size)

    ext = _load_ext()
    if (ext is not None and windows.is_cuda and windows.dtype == torch.float16 and C % 8 == 0):
        x = ext.window_unpartition_fwd(windows.contiguous(), window_size, Hp, Wp)
    else:
        # Triton fallback
        try:
            from app.core.vendor.sam_triton_kernels import triton_window_unpartition
            x = triton_window_unpartition(windows, window_size, (Hp, Wp))
        except Exception:
            # PyTorch fallback
            x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

    # Crop padding back to original size (same as SAM's original)
    if hw is not None:
        H, W = hw
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
    return x


def leaky_relu_scale_add_cuda(
    x: torch.Tensor,
    residual: torch.Tensor,
    neg_slope: float = 0.2,
    scale: float = 0.2,
) -> torch.Tensor:
    """Fused LeakyReLU(x)*scale + residual — ESRGAN RRDB skip connection."""
    ext = _load_ext()
    if (ext is not None and x.is_cuda and x.dtype == torch.float16
            and x.numel() % 2 == 0):
        return ext.leaky_relu_scale_add(
            x.contiguous(), residual.contiguous(), neg_slope, scale
        )
    # Triton fallback
    try:
        from app.core.vendor.esrgan_triton_kernels import triton_leakyrelu_inplace, triton_scale_add
        lrelu = triton_leakyrelu_inplace(x.clone(), neg_slope)
        return triton_scale_add(lrelu, residual, scale)
    except Exception:
        pass
    # PyTorch fallback
    return F.leaky_relu(x, negative_slope=neg_slope) * scale + residual


def pixel_shuffle_2x_cuda(x: torch.Tensor) -> torch.Tensor:
    """
    Pixel shuffle 2× — delegates to F.pixel_shuffle.

    Our custom CUDA kernel (pixel_shuffle_2x in corbeau_kernels.cu) uses scattered
    NCHW reads that perform no better than PyTorch's highly optimised
    view+permute+contiguous path.  Using F.pixel_shuffle avoids the overhead of
    the extension dispatch while achieving the same speed.
    """
    return F.pixel_shuffle(x, 2)


def ext_available() -> bool:
    """Return True if the CUDA C++ extension compiled successfully."""
    return _load_ext() is not None
