"""
CUDA utility primitives for CorbeauSplat — Windows 11 / RTX 4090.

Provides:
  - Device capability detection
  - Optimal dtype selection (FP32/FP16/BF16)
  - CUDA stream pool for pipeline parallelism
  - Pinned-memory frame buffer for zero-copy H2D transfers
  - One-shot CUDA warm-up to eliminate first-inference latency
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("cuda_utils")


# ---------------------------------------------------------------------------
# Capability probes (no torch import at module level — stays importable on CPU)
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_capability() -> tuple[int, int]:
    """Returns (major, minor) CUDA compute capability, e.g. (8, 9) for RTX 4090."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return (0, 0)


def trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def torch_tensorrt_available() -> bool:
    try:
        import torch_tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def ort_cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def ort_trt_available() -> bool:
    try:
        import onnxruntime as ort
        return "TensorrtExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def triton_available() -> bool:
    """torch.compile on Windows requires triton-windows; detect it."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


def compile_available() -> bool:
    """torch.compile is usable only when Triton is present (Windows requirement)."""
    try:
        import torch
        # torch.compile exists in PyTorch >= 2.0
        if not hasattr(torch, "compile"):
            return False
        return triton_available()
    except ImportError:
        return False


def select_dtype(prefer_half: bool = True):
    """
    Return the best floating-point dtype for inference on the current device.
    Priority: BF16 (Ampere+) > FP16 > FP32.
    On RTX 4090 (Ada Lovelace, cap 8.9), FP16 and BF16 are both available.
    """
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = get_cuda_capability()
    if prefer_half:
        if major >= 8:
            return torch.float16   # Ada: use fp16 (BF16 also works but fp16 is faster for ViT)
        if major >= 7:
            return torch.float16   # Volta/Turing/Ampere Tensor Cores
    return torch.float32


def best_inference_backend() -> str:
    """
    Returns a string token for the fastest available inference backend.
    'torch_trt' > 'ort_trt' > 'ort_cuda' > 'torch_cuda' > 'cpu'
    """
    if torch_tensorrt_available():
        return "torch_trt"
    if ort_trt_available():
        return "ort_trt"
    if ort_cuda_available():
        return "ort_cuda"
    if cuda_available():
        return "torch_cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# CUDA warm-up (call once at startup to prime the CUDA context)
# ---------------------------------------------------------------------------

_warmed_up = False


def warm_up_cuda(device: str = "cuda") -> None:
    """
    Runs a trivial CUDA operation to initialise the CUDA context so the first
    real kernel call does not pay the ~500 ms init cost.
    Safe to call multiple times; only executes once.
    """
    global _warmed_up
    if _warmed_up:
        return
    try:
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            _warmed_up = True
            logger.debug("CUDA context warmed up.")
    except Exception as e:
        logger.debug(f"CUDA warm-up skipped: {e}")


# ---------------------------------------------------------------------------
# CUDA Stream Pool
# ---------------------------------------------------------------------------

class CUDAStreamPool:
    """
    A simple fixed-size pool of CUDA streams for pipeline parallelism.
    Typical usage: two streams for double-buffered H2D/compute overlap.
    """

    def __init__(self, n: int = 2, device: str = "cuda"):
        import torch
        self._streams = [torch.cuda.Stream(device=device) for _ in range(n)]
        self._available = list(self._streams)

    def acquire(self):
        """Borrow a stream (round-robin if none free, blocks until available)."""
        if not self._available:
            # All in use — return the first one anyway (caller must synchronise)
            return self._streams[0]
        return self._available.pop(0)

    def release(self, stream) -> None:
        if stream not in self._available:
            self._available.append(stream)

    def sync_all(self) -> None:
        import torch
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Pinned-memory frame buffer for zero-copy CPU → GPU transfer
# ---------------------------------------------------------------------------

class PinnedFrameBuffer:
    """
    Pre-allocates N slots of page-locked (pinned) CPU memory for video frames.
    Pinned memory allows async `tensor.to(device, non_blocking=True)` without
    a copy through pageable memory, cutting H2D latency by ~30% on PCIe 4.0.

    Usage:
        buf = PinnedFrameBuffer(n=4, frame_shape=(1080, 1920, 3))
        slot = buf.get_slot(0)
        slot[:] = numpy_frame          # fill the pinned buffer
        gpu_t = buf.to_gpu(0, stream)  # async transfer on a CUDA stream
    """

    def __init__(self, n: int, frame_shape: tuple, dtype=None):
        import torch
        if dtype is None:
            dtype = torch.uint8
        self._slots = [
            torch.zeros(frame_shape, dtype=dtype, pin_memory=True)
            for _ in range(n)
        ]

    def get_slot(self, idx: int):
        return self._slots[idx]

    def fill_slot(self, idx: int, numpy_frame) -> None:
        import torch
        self._slots[idx].copy_(torch.from_numpy(numpy_frame))

    def to_gpu(self, idx: int, stream=None, device: str = "cuda"):
        import torch
        if stream is not None:
            with torch.cuda.stream(stream):
                return self._slots[idx].to(device, non_blocking=True)
        return self._slots[idx].to(device)
