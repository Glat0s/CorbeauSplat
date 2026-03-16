"""
Optimized Real-ESRGAN super-resolution for CorbeauSplat — RTX 4090.

This version uses ONNX Runtime (with TensorRT or CUDA) for high-performance
inference. It does NOT depend on basicsr or realesrgan packages.

Key features:
  1. Preferred ONNX model path (RealESRGAN_x4plus.fp16.onnx).
  2. TensorRT optimization for maximum throughput on RTX GPUs.
  3. Seamless tiling for large images to avoid OOM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("esrgan_optimized")

# ---------------------------------------------------------------------------
# Optimized engine
# ---------------------------------------------------------------------------


class OptimizedRealESRGAN:
    """
    High-throughput Real-ESRGAN upscaler optimised for Windows 11 / RTX 4090.
    Uses ONNX Runtime with TensorRT.

    Parameters
    ----------
    model_path : str | Path
        Path to the Real-ESRGAN model (can be pth or onnx, but onnx is preferred).
    scale : int
        Upscale factor (default 4).
    tile : int
        Tile size for memory-limited processing (default 512).
    tile_pad : int
        Tile padding to avoid seam artefacts (default 10).
    device : str
        ``"cuda"`` or ``"cpu"``.
    batch_size : int
        Number of tiles / images to process per GPU batch (not yet fully implemented for ONNX).
    """

    def __init__(
        self,
        model_path: str | Path = "",
        scale: int = 4,
        tile: int = 512,
        tile_pad: int = 10,
        device: str = "cuda",
        use_fp16: bool = True,
        use_compile: bool = True,  # Ignored in ONNX path
        batch_size: int = 4,
    ):
        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self._device = device
        self._use_fp16 = use_fp16 and device == "cuda"
        self.batch_size = batch_size
        self._ort_session = None
        self._ort_input_name = None

        if model_path:
            self.load(model_path)

    # ------------------------------------------------------------------

    def load(self, model_path: str | Path) -> bool:
        """Load model weights; returns True on success."""
        # --- Check for preferred ONNX model in the app weights directory ---
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        external_onnx = weights_dir / "RealESRGAN_x4plus.fp16.onnx"
        if external_onnx.exists():
            logger.info("Preferred ESRGAN ONNX model found: %s", external_onnx)
            if self.build_trt_session(external_onnx):
                logger.info("ESRGAN using TensorRT ONNX session.")
                return True

        model_path = Path(model_path)
        if (
            model_path.exists()
            and model_path.suffix.lower() == ".onnx"
            and self.build_trt_session(model_path)
        ):
            return True

        logger.warning("No suitable ESRGAN ONNX model found and basicsr/realesrgan are disabled.")
        return False

    def build_trt_session(self, onnx_path: Path, trt_cache_dir: Optional[Path] = None) -> bool:
        """
        Build an ORT InferenceSession with TensorrtExecutionProvider.
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            return False

        if trt_cache_dir is None:
            trt_cache_dir = onnx_path.parent / "trt_cache"
        trt_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            import onnxruntime as ort

            providers = []
            if "TensorrtExecutionProvider" in ort.get_available_providers():
                providers = [
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": str(trt_cache_dir),
                            "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
                        },
                    ),
                    "CUDAExecutionProvider",
                ]
            elif "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider"]
            else:
                logger.warning("No GPU ORT provider available.")
                return False

            self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            logger.info(
                "ORT session ready (providers: %s).",
                [p[0] if isinstance(p, tuple) else p for p in providers],
            )
            return True

        except Exception as e:
            logger.error("build_trt_session failed: %s", e)
            return False

    def upscale_image_trt(self, bgr: np.ndarray, outscale: float = 4.0) -> Optional[np.ndarray]:
        """
        Run ORT TRT/CUDA inference for a single BGR image.
        """
        if self._ort_session is None:
            return None

        try:
            import cv2

            h, w = bgr.shape[:2]

            # Simple tiling if needed (rudimentary implementation)
            if self.tile > 0 and (h > self.tile or w > self.tile):
                return self._upscale_tiled(bgr, outscale)

            rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
            t = rgb.transpose(2, 0, 1)[np.newaxis]  # (1,3,H,W) float32

            result = self._ort_session.run(None, {self._ort_input_name: t})[0]  # (1,3,H*s,W*s)
            out = result[0].transpose(1, 2, 0).clip(0, 1)
            out_u8 = (out * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR

            if abs(outscale - self.scale) > 0.01:
                new_h = int(h * outscale)
                new_w = int(w * outscale)
                out_u8 = cv2.resize(out_u8, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            return out_u8
        except Exception as e:
            logger.error("upscale_image_trt error: %s", e)
            return None

    def _upscale_tiled(self, bgr: np.ndarray, outscale: float) -> np.ndarray:
        """Rudimentary tiling for ONNX path to avoid OOM."""
        import cv2

        h, w = bgr.shape[:2]
        tile = self.tile
        pad = self.tile_pad
        output_h = int(h * outscale)
        output_w = int(w * outscale)
        output = np.zeros((output_h, output_w, 3), dtype=np.uint8)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                # Extract tile with padding
                y1 = max(0, y - pad)
                x1 = max(0, x - pad)
                y2 = min(h, y + tile + pad)
                x2 = min(w, x + tile + pad)

                img_tile = bgr[y1:y2, x1:x2]

                # Inference on tile
                rgb_tile = img_tile[:, :, ::-1].astype(np.float32) / 255.0
                t = rgb_tile.transpose(2, 0, 1)[np.newaxis]
                res_tile = self._ort_session.run(None, {self._ort_input_name: t})[0][0]
                res_tile = res_tile.transpose(1, 2, 0).clip(0, 1)
                res_tile = (res_tile * 255).astype(np.uint8)[:, :, ::-1]

                # Rescale if needed
                if abs(outscale - self.scale) > 0.01:
                    th = int((y2 - y1) * outscale)
                    tw = int((x2 - x1) * outscale)
                    res_tile = cv2.resize(res_tile, (tw, th), interpolation=cv2.INTER_LANCZOS4)

                # Paste into output, removing padding
                oy1 = int(y * outscale)
                oy2 = int(min(h, y + tile) * outscale)
                ox1 = int(x * outscale)
                ox2 = int(min(w, x + tile) * outscale)

                py1 = int((y - y1) * outscale)
                px1 = int((x - x1) * outscale)
                py2 = py1 + (oy2 - oy1)
                px2 = px1 + (ox2 - ox1)

                output[oy1:oy2, ox1:ox2] = res_tile[py1:py2, px1:px2]

        return output

    def upscale_image(self, bgr: np.ndarray, outscale: float = 4.0) -> Optional[np.ndarray]:
        """
        Upscale a single BGR uint8 image.
        """
        if self._ort_session is not None:
            return self.upscale_image_trt(bgr, outscale)
        logger.warning("OptimizedRealESRGAN not loaded — cannot upscale.")
        return None

    def upscale_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        outscale: float = 4.0,
        progress_callback=None,
        check_cancel=None,
    ) -> int:
        """
        Upscale all images in *input_dir* and write to *output_dir*.
        """
        if self._ort_session is None:
            logger.warning("ESRGAN not loaded — skipping upscale_folder.")
            return 0

        import cv2

        output_dir.mkdir(parents=True, exist_ok=True)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in exts)
        total = len(image_paths)
        if total == 0:
            return 0

        done = 0
        for i, img_path in enumerate(image_paths):
            if check_cancel and check_cancel():
                break
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            result = self.upscale_image(bgr, outscale=outscale)
            if result is not None:
                out_path = output_dir / (img_path.stem + ".png")
                cv2.imwrite(str(out_path), result)
                done += 1
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))
        return done

    def unload(self) -> None:
        """Release session."""
        self._ort_session = None
        self._ort_input_name = None
