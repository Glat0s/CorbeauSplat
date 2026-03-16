#!/usr/bin/env python3
"""
CorbeauSplat GPU Inference Benchmarks
======================================
Tests all inference backends on Windows 11 / RTX 4090 (CUDA 12.9 / torch 2.8.0).

Usage:
    python benchmarks/benchmark_inference.py [--runs N] [--warmup N] [--device cuda]

Output: Markdown table printed to stdout + saved to benchmarks/results.md
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────

def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def timeit(fn: Callable, runs: int = 20, warmup: int = 3) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) after *warmup* un-timed calls."""
    for _ in range(warmup):
        fn()
    _cuda_sync()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────
# Benchmark suites
# ─────────────────────────────────────────────────────────────────

def bench_chroma_key(runs: int, warmup: int, device: str) -> list[dict]:
    """GPUChromaKey: CPU OpenCV vs GPU PyTorch."""
    results = []
    try:
        from app.core.gpu_chroma_key import GPUChromaKey
        dummy = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # CPU
        ck_cpu = GPUChromaKey(device="cpu")
        mean, std = timeit(lambda: ck_cpu.process_frame(dummy), runs, warmup)
        results.append({"name": "ChromaKey CPU (OpenCV)", "mean_ms": mean, "std_ms": std})

        # GPU
        if device == "cuda":
            ck_gpu = GPUChromaKey(device="cuda")
            mean, std = timeit(lambda: ck_gpu.process_frame(dummy), runs, warmup)
            results.append({"name": "ChromaKey GPU (PyTorch+kornia)", "mean_ms": mean, "std_ms": std})

            # GPU batch=16
            frames = [dummy] * 16
            mean, std = timeit(lambda: ck_gpu.process_batch(frames), runs, warmup)
            results.append({"name": "ChromaKey GPU batch=16 (per-frame)", "mean_ms": mean / 16, "std_ms": std / 16})

    except Exception as e:
        results.append({"name": "ChromaKey", "mean_ms": -1, "std_ms": 0, "error": str(e)})
    return results


def bench_esrgan(runs: int, warmup: int, device: str) -> list[dict]:
    """RealESRGAN: PyTorch eager vs FP16 vs torch.compile vs ORT CUDA vs ORT TRT."""
    results = []
    weights_dir = ROOT / "app" / "weights"
    model_path = weights_dir / "RealESRGAN_x4plus.pth"

    if not model_path.exists():
        results.append({"name": "ESRGAN (weights missing)", "mean_ms": -1, "std_ms": 0})
        return results

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    try:
        from app.core.esrgan_optimized import OptimizedRealESRGAN

        # PyTorch FP16 + torch.compile
        esrgan = OptimizedRealESRGAN(model_path=model_path, device=device, use_fp16=True, use_compile=True)
        if esrgan.load(model_path):
            mean, std = timeit(lambda: esrgan.upscale_image(dummy), runs, warmup)
            results.append({"name": "ESRGAN PyTorch FP16+compile", "mean_ms": mean, "std_ms": std})
            esrgan.unload()

        # ORT CUDA EP
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                esrgan2 = OptimizedRealESRGAN(model_path=model_path, device=device, use_fp16=False, use_compile=False)
                if esrgan2.load(model_path):
                    onnx_path = weights_dir / "realesrgan.onnx"
                    if not onnx_path.exists():
                        esrgan2.export_onnx(onnx_path)
                    if onnx_path.exists() and esrgan2.build_trt_session(onnx_path):
                        mean, std = timeit(lambda: esrgan2.upscale_image_trt(dummy), runs, warmup)
                        results.append({"name": "ESRGAN ORT GPU EP", "mean_ms": mean, "std_ms": std})
                    esrgan2.unload()
        except Exception as e:
            results.append({"name": "ESRGAN ORT GPU EP", "mean_ms": -1, "std_ms": 0, "error": str(e)})

    except Exception as e:
        results.append({"name": "ESRGAN", "mean_ms": -1, "std_ms": 0, "error": str(e)})

    return results


def bench_sam(runs: int, warmup: int, device: str) -> list[dict]:
    """SAM: eager vs torch.compile vs CUDA graph encoder."""
    results = []
    ckpt = ROOT / "engines" / "sam_vit_b_01ec64.pth"
    if not ckpt.exists():
        results.append({"name": "SAM (checkpoint missing)", "mean_ms": -1, "std_ms": 0})
        return results

    dummy_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    try:
        from app.core.sam_optimized import PersistentSAMPredictor

        # Eager
        sam_eager = PersistentSAMPredictor(ckpt, model_type="vit_b", device=device, use_fp16=True, use_compile=False)
        if sam_eager.is_loaded:
            mean, std = timeit(lambda: sam_eager.predict_frame(dummy_rgb), runs // 2, warmup)
            results.append({"name": "SAM vit_b eager FP16", "mean_ms": mean, "std_ms": std})
            sam_eager.unload()

        # torch.compile + CUDA graph
        sam_opt = PersistentSAMPredictor(ckpt, model_type="vit_b", device=device, use_fp16=True, use_compile=True)
        if sam_opt.is_loaded:
            mean, std = timeit(lambda: sam_opt.predict_frame(dummy_rgb), runs // 2, warmup)
            results.append({"name": "SAM vit_b compile+CUDA graph", "mean_ms": mean, "std_ms": std})
            sam_opt.unload()

    except Exception as e:
        results.append({"name": "SAM", "mean_ms": -1, "std_ms": 0, "error": str(e)})

    return results


def bench_gfpgan(runs: int, warmup: int, device: str) -> list[dict]:
    """GFPGAN: PyTorch FP32 vs FP16+Triton vs CUDA graph."""
    results = []
    ckpt = ROOT / "app" / "weights" / "GFPGANv1.4.pth"
    if not ckpt.exists():
        results.append({"name": "GFPGAN (weights missing)", "mean_ms": -1, "std_ms": 0})
        return results

    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    try:
        from app.core.gfpgan_engine import GFPGANEngine

        # FP16 Triton (no CUDA graph)
        gfpgan_t2 = GFPGANEngine(checkpoint=ckpt, device=device, use_fp16=True, use_cuda_graph=False)
        if gfpgan_t2.load():
            mean, std = timeit(lambda: gfpgan_t2.enhance_frame(dummy), runs, warmup)
            results.append({"name": "GFPGAN Triton FP16 (Tier 2)", "mean_ms": mean, "std_ms": std})
            gfpgan_t2.unload()

        # FP16 Triton + CUDA graph (Tier 3)
        gfpgan_t3 = GFPGANEngine(checkpoint=ckpt, device=device, use_fp16=True, use_cuda_graph=True)
        if gfpgan_t3.load():
            mean, std = timeit(lambda: gfpgan_t3.enhance_frame(dummy), runs, warmup)
            results.append({"name": "GFPGAN Triton+CUDA graph (Tier 3)", "mean_ms": mean, "std_ms": std})
            gfpgan_t3.unload()

    except Exception as e:
        results.append({"name": "GFPGAN", "mean_ms": -1, "std_ms": 0, "error": str(e)})

    return results


def bench_xseg(runs: int, warmup: int, device: str) -> list[dict]:
    """XSeg vs SAM for segmentation."""
    results = []
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    xseg_ckpt = ROOT / "app" / "weights" / "XSeg_model.pth"

    try:
        from app.core.xseg_engine import XSegEngine
        if xseg_ckpt.exists():
            xseg = XSegEngine(checkpoint=xseg_ckpt, device=device, use_fp16=True, use_cuda_graph=True)
            if xseg.load():
                mean, std = timeit(lambda: xseg.predict_frame(dummy), runs, warmup)
                results.append({"name": "XSeg FP16+CUDA graph", "mean_ms": mean, "std_ms": std})
                xseg.unload()
        else:
            results.append({"name": "XSeg (weights missing)", "mean_ms": -1, "std_ms": 0})
    except Exception as e:
        results.append({"name": "XSeg", "mean_ms": -1, "std_ms": 0, "error": str(e)})

    return results


# ─────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────

def format_table(sections: dict[str, list[dict]]) -> str:
    lines = []
    lines.append("| Benchmark | Mean (ms) | Std (ms) | Notes |")
    lines.append("|-----------|-----------|----------|-------|")
    for section, rows in sections.items():
        lines.append(f"| **{section}** | | | |")
        for r in rows:
            if r["mean_ms"] < 0:
                note = r.get("error", "skipped")
                lines.append(f"| {r['name']} | — | — | {note} |")
            else:
                lines.append(f"| {r['name']} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CorbeauSplat inference benchmarks")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed runs per benchmark")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    print(f"CorbeauSplat Inference Benchmarks")
    print(f"Device: {args.device} | Runs: {args.runs} | Warmup: {args.warmup}")
    print()

    if args.device == "cuda":
        try:
            import torch
            from app.core.cuda_utils import warm_up_cuda
            warm_up_cuda()
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
        except Exception:
            pass
    print()

    sections = {
        "Chroma Key (1920×1080)": bench_chroma_key(args.runs, args.warmup, args.device),
        "RealESRGAN (256×256 → 1024×1024)": bench_esrgan(args.runs, args.warmup, args.device),
        "SAM (512×512 face)": bench_sam(args.runs, args.warmup, args.device),
        "GFPGAN (512×512 face)": bench_gfpgan(args.runs, args.warmup, args.device),
        "XSeg (512×512 → 256×256 mask)": bench_xseg(args.runs, args.warmup, args.device),
    }

    table = format_table(sections)
    print(table)

    # Save results
    out = ROOT / "benchmarks" / "results.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# CorbeauSplat Inference Benchmark Results\n\n")
        f.write(f"**Platform:** Windows 11 / RTX 4090 / CUDA 12.9 / PyTorch 2.8.0+cu129  \n")
        f.write(f"**Runs:** {args.runs} | **Warmup:** {args.warmup}  \n\n")
        f.write(table)
        f.write("\n\n*Generated by `benchmarks/benchmark_inference.py`*\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
