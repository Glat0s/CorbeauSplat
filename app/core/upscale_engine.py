import logging
from pathlib import Path

from .base_engine import BaseEngine
from .esrgan_optimized import OptimizedRealESRGAN

logger = logging.getLogger("upscale_engine")


class UpscaleEngine(BaseEngine):
    """
    Engine for Real-ESRGAN upscaling using ONNX Runtime.
    Handles high-performance inference via TensorRT/CUDA.
    """

    _MODEL_URLS = {
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    }

    def __init__(self, logger_callback=None):
        super().__init__("Upscale", logger_callback)
        self._gfpgan = None

    def is_installed(self):
        """Checks if onnxruntime is importable."""
        import importlib.util

        return importlib.util.find_spec("onnxruntime") is not None

    def get_version(self):
        """Returns installed version of onnxruntime"""
        try:
            import onnxruntime

            return getattr(onnxruntime, "__version__", "Unknown")
        except ImportError:
            return None

    def get_models_path(self) -> Path:
        """Returns the directory where weights are stored"""
        # We store weights in app/weights for persistence
        root = Path(__file__).resolve().parent.parent
        weights_dir = root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        return weights_dir

    def check_model_availability(self, model_name):
        """Checks if model weights are present locally."""
        weights_dir = self.get_models_path()
        # Accept either .pth or .onnx variants
        for ext in (".pth", ".onnx", ".fp16.onnx"):
            if (weights_dir / f"{model_name}{ext}").exists():
                return True
        return False

    def download_model(self, model_name: str) -> bool:
        """Download model weights from the official Real-ESRGAN release."""
        url = self._MODEL_URLS.get(model_name)
        if not url:
            self.log(f"No download URL for model: {model_name}")
            return False
        dest = self.get_models_path() / f"{model_name}.pth"
        try:
            import urllib.request

            self.log(f"Downloading {model_name} from {url} …")
            urllib.request.urlretrieve(url, str(dest))
            self.log(f"Downloaded to {dest}")
            return True
        except Exception as e:
            self.log(f"Download failed: {e}")
            return False

    def load_model(self, model_name="RealESRGAN_x4plus", tile=512, target_scale=4, half=True):
        """
        Loads the OptimizedRealESRGAN model.
        Returns the model object or None.
        """
        if not self.is_installed():
            self.log("onnxruntime not installed.")
            return None

        try:
            # We use the optimized engine which handles ONNX/TRT
            upsampler = OptimizedRealESRGAN(
                scale=4,  # Internal model scale is usually 4
                tile=tile,
                tile_pad=10,
                device=self.device,
                use_fp16=half,
            )

            # The .load() method checks for the preferred external ONNX first
            if upsampler.load(""):  # Passing empty string to let it check external_onnx
                # Inject target scale for later resizing if needed
                upsampler.target_scale = target_scale
                return upsampler
            else:
                self.log("Failed to load ESRGAN ONNX model.")
                return None

        except Exception as e:
            self.log(f"Failed to load model: {e}")
            return None

    def upscale_image(self, input_path, output_path, upsampler, face_enhance=False):
        """
        Upscales a single image using the OptimizedRealESRGAN instance.
        """
        try:
            import cv2

            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                self.log(f"Failed to read {input_path}")
                return False

            final_scale = 4
            if hasattr(upsampler, "target_scale") and upsampler.target_scale:
                final_scale = upsampler.target_scale

            output = upsampler.upscale_image(img, outscale=final_scale)

            if output is None:
                self.log(f"Upscaling failed for {input_path}")
                return False

            # Face Enhance (GFPGAN)
            if face_enhance:
                try:
                    from app.core.gfpgan_engine import GFPGANEngine

                    if not hasattr(self, "_gfpgan") or self._gfpgan is None:
                        self._gfpgan = GFPGANEngine(
                            device=self.device if hasattr(self, "device") else "cuda"
                        )
                        self._gfpgan.load()
                    if self._gfpgan.is_loaded:
                        output = self._gfpgan.enhance_frame(output)
                except Exception as _e:
                    self.log(f"GFPGAN face enhancement skipped: {_e}")

            cv2.imwrite(output_path, output)
            return True
        except Exception as e:
            self.log(f"Error upscaling {input_path}: {e}")
            return False

    def upscale_folder(
        self,
        input_dir,
        output_dir,
        extension="jpg",
        model_name="RealESRGAN_x4plus",
        tile=512,
        target_scale=4,
        face_enhance=False,
    ):
        """
        Upscales all images in input_dir to output_dir.
        """
        in_p = Path(input_dir)
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        files = sorted(
            [
                f
                for f in in_p.iterdir()
                if f.is_file()
                and f.suffix.lower() in (f".{extension}", ".png", ".jpg", ".jpeg", ".webp")
            ]
        )

        self.log(f"Found {len(files)} images to upscale.")

        upsampler = self.load_model(model_name=model_name, tile=tile, target_scale=target_scale)
        if not upsampler:
            return False, "Failed to load model"

        success_count = 0
        for idx, img_path in enumerate(files):
            self.log(f"Upscaling [{idx+1}/{len(files)}]: {img_path.name} (x{target_scale})")
            if self.upscale_image(
                str(img_path),
                str(out_p / (img_path.stem + ".png")),
                upsampler,
                face_enhance=face_enhance,
            ):
                success_count += 1

        self.log(f"Upscaling complete. {success_count}/{len(files)} processed.")
        return True, "Success"
