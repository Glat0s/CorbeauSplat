import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.core.esrgan_kernels import inject_esrgan_kernels  # noqa: E402
from app.core.esrgan_optimized import OptimizedRealESRGAN  # noqa: E402
from app.core.sam_kernels import inject_sam_triton_kernels  # noqa: E402
from app.core.sam_optimized import PersistentSAMPredictor  # noqa: E402


def profile_model(model_name, model, input_data, runs=20, warmup=5):
    # Warmup
    for _ in range(warmup):
        model(input_data)
    torch.cuda.synchronize()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(input_data)

    print(f"\n=== Profile for {model_name} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    device = "cuda"

    # SAM Profile
    sam_ckpt = ROOT / "engines" / "sam_vit_b_01ec64.pth"
    if sam_ckpt.exists():
        sam_predictor = PersistentSAMPredictor(
            sam_ckpt, "vit_b", device=device, use_fp16=True, use_compile=True
        )
        inject_sam_triton_kernels(sam_predictor._sam.image_encoder)

        dummy_rgb = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.float16)

        # Profile image_encoder
        profile_model(
            "SAM Image Encoder (with Triton)", sam_predictor._sam.image_encoder, dummy_rgb
        )
    else:
        print("SAM checkpoint missing, skipping profile.")

    # ESRGAN Profile
    esrgan_path = ROOT / "app" / "weights" / "RealESRGAN_x4plus.pth"
    if esrgan_path.exists():
        esrgan = OptimizedRealESRGAN(
            model_path=esrgan_path, device=device, use_fp16=True, use_compile=True
        )
        esrgan.load(esrgan_path)
        inject_esrgan_kernels(esrgan._upsampler.model)

        dummy_img = torch.randn(1, 3, 256, 256, device=device, dtype=torch.float16)
        profile_model("ESRGAN (with Triton)", esrgan._upsampler.model, dummy_img)
    else:
        print("ESRGAN weights missing, skipping profile.")


if __name__ == "__main__":
    main()
