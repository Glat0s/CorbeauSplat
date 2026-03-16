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
# Standalone Triton kernel micro-benchmarks (no model weights needed)
# ─────────────────────────────────────────────────────────────────

def bench_triton_layernorm(runs: int, warmup: int, device: str) -> list[dict]:
    """TritonLayerNorm vs nn.LayerNorm on typical SAM hidden dimensions."""
    results = []
    if device != "cuda":
        return results
    try:
        import torch
        import torch.nn as nn
        from app.core.vendor.sam_triton_kernels import TritonLayerNorm

        for dim in (768, 1024, 1280):  # ViT-B / ViT-L / ViT-H embed dims
            # Batch of 4096 tokens (64×64 feature map, 1 image)
            x = torch.randn(4096, dim, device=device, dtype=torch.float16)

            # Baseline: PyTorch nn.LayerNorm
            ln_pt = nn.LayerNorm(dim, eps=1e-6).to(device).half()
            mean, std = timeit(lambda: ln_pt(x), runs, warmup)
            results.append({"name": f"LayerNorm-{dim} PyTorch", "mean_ms": mean, "std_ms": std})

            # Triton single-pass
            ln_tri = TritonLayerNorm(dim, eps=1e-6).to(device)
            ln_tri.weight.data.copy_(ln_pt.weight.data)
            ln_tri.bias.data.copy_(ln_pt.bias.data)
            mean, std = timeit(lambda: ln_tri(x), runs, warmup)
            results.append({"name": f"LayerNorm-{dim} Triton (single-pass)", "mean_ms": mean, "std_ms": std})

            # Triton + fused GELU
            ln_gelu = TritonLayerNorm(dim, eps=1e-6, fuse_gelu=True).to(device)
            ln_gelu.weight.data.copy_(ln_pt.weight.data)
            ln_gelu.bias.data.copy_(ln_pt.bias.data)
            mean, std = timeit(lambda: ln_gelu(x), runs, warmup)
            results.append({"name": f"LayerNorm+GELU-{dim} Triton (fused)", "mean_ms": mean, "std_ms": std})

    except Exception as e:
        results.append({"name": "Triton LayerNorm", "mean_ms": -1, "std_ms": 0, "error": str(e)})
    return results


def bench_triton_window_ops(runs: int, warmup: int, device: str) -> list[dict]:
    """Triton window partition/unpartition vs PyTorch permute+reshape."""
    results = []
    if device != "cuda":
        return results
    try:
        import torch
        from app.core.vendor.sam_triton_kernels import triton_window_partition, triton_window_unpartition

        # Typical SAM ViT-B feature map: (1, 64, 64, 768)
        B, H, W, C = 1, 64, 64, 768
        x = torch.randn(B, H, W, C, device=device, dtype=torch.float16)
        window_size = 14  # SAM default

        # PyTorch baseline (SAM's original implementation)
        def pt_partition(t):
            t2 = t.view(B, H // window_size, window_size, W // window_size, window_size, C)
            return t2.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

        mean, std = timeit(lambda: pt_partition(x), runs, warmup)
        results.append({"name": "Window partition PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_window_partition(x, window_size), runs, warmup)
        results.append({"name": "Window partition Triton", "mean_ms": mean, "std_ms": std})

        windows, hw = triton_window_partition(x, window_size)
        mean, std = timeit(lambda: triton_window_unpartition(windows, window_size, hw), runs, warmup)
        results.append({"name": "Window unpartition Triton", "mean_ms": mean, "std_ms": std})

    except Exception as e:
        results.append({"name": "Triton window ops", "mean_ms": -1, "std_ms": 0, "error": str(e)})
    return results


def bench_triton_esrgan_ops(runs: int, warmup: int, device: str) -> list[dict]:
    """ESRGAN Triton ops: scale_add, leakyrelu, pixel_shuffle, MemEfficientDenseBlock."""
    results = []
    if device != "cuda":
        return results
    try:
        import torch
        import torch.nn.functional as F
        from app.core.vendor.esrgan_triton_kernels import (
            triton_scale_add, triton_leakyrelu_inplace,
            triton_pixel_shuffle_2x, MemEfficientDenseBlock,
        )

        # --- scale_add (fused x*0.2 + residual) ---
        # Typical RRDB output: (1, 64, 128, 128)
        x  = torch.randn(1, 64, 128, 128, device=device, dtype=torch.float16)
        r  = torch.randn_like(x)

        mean, std = timeit(lambda: x * 0.2 + r, runs, warmup)
        results.append({"name": "scale_add PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_scale_add(x, r, 0.2), runs, warmup)
        results.append({"name": "scale_add Triton (fused)", "mean_ms": mean, "std_ms": std})

        # --- leakyrelu_inplace ---
        a = torch.randn(1, 64, 128, 128, device=device, dtype=torch.float16)

        mean, std = timeit(lambda: F.leaky_relu_(a.clone(), negative_slope=0.2), runs, warmup)
        results.append({"name": "LeakyReLU PyTorch inplace", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_leakyrelu_inplace(a.clone(), 0.2), runs, warmup)
        results.append({"name": "LeakyReLU Triton inplace", "mean_ms": mean, "std_ms": std})

        # --- pixel_shuffle 2× ---
        # After last ESRGAN conv: (1, 256, 256, 256) → (1, 64, 512, 512)
        ps_in = torch.randn(1, 256, 256, 256, device=device, dtype=torch.float16)

        mean, std = timeit(lambda: F.pixel_shuffle(ps_in, 2), runs, warmup)
        results.append({"name": "PixelShuffle-2× PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_pixel_shuffle_2x(ps_in), runs, warmup)
        results.append({"name": "PixelShuffle-2× Triton", "mean_ms": mean, "std_ms": std})

        # --- MemEfficientDenseBlock vs cat-based forward ---
        try:
            from basicsr.archs.rrdbnet_arch import ResidualDenseBlock
            rdb_orig = ResidualDenseBlock(num_feat=64, num_grow_ch=32).to(device).half()
            rdb_eff  = MemEfficientDenseBlock.from_module(rdb_orig).to(device).half()
            feat = torch.randn(1, 64, 64, 64, device=device, dtype=torch.float16)

            mean, std = timeit(lambda: rdb_orig(feat), runs, warmup)
            results.append({"name": "DenseBlock (basicsr, cat-based)", "mean_ms": mean, "std_ms": std})

            mean, std = timeit(lambda: rdb_eff(feat), runs, warmup)
            results.append({"name": "DenseBlock (MemEfficient, pre-alloc)", "mean_ms": mean, "std_ms": std})
        except ImportError:
            results.append({"name": "DenseBlock (basicsr missing)", "mean_ms": -1, "std_ms": 0})

    except Exception as e:
        results.append({"name": "Triton ESRGAN ops", "mean_ms": -1, "std_ms": 0, "error": str(e)})
    return results


def bench_sam_with_triton(runs: int, warmup: int, device: str) -> list[dict]:
    """SAM vit_b end-to-end: baseline vs Triton kernel injection."""
    results = []
    ckpt = ROOT / "engines" / "sam_vit_b_01ec64.pth"
    if not ckpt.exists():
        results.append({"name": "SAM+Triton (checkpoint missing)", "mean_ms": -1, "std_ms": 0})
        return results

    dummy_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    try:
        import torch
        from app.core.sam_optimized import PersistentSAMPredictor
        from app.core.sam_kernels import inject_sam_triton_kernels

        # Baseline: compile + CUDA graph (no Triton LayerNorm/window ops)
        sam_base = PersistentSAMPredictor(ckpt, "vit_b", device=device, use_fp16=True, use_compile=True)
        if sam_base.is_loaded:
            mean, std = timeit(lambda: sam_base.predict_frame(dummy_rgb), runs // 2, warmup)
            results.append({"name": "SAM vit_b compile+CUDA graph (baseline)", "mean_ms": mean, "std_ms": std})

        # +Triton LayerNorm + window partition injection
        sam_tri = PersistentSAMPredictor(ckpt, "vit_b", device=device, use_fp16=True, use_compile=True)
        if sam_tri.is_loaded:
            inject_sam_triton_kernels(sam_tri._sam.image_encoder)
            mean, std = timeit(lambda: sam_tri.predict_frame(dummy_rgb), runs // 2, warmup)
            results.append({"name": "SAM vit_b +Triton LayerNorm+WindowOps", "mean_ms": mean, "std_ms": std})
            sam_tri.unload()

        if sam_base.is_loaded:
            sam_base.unload()

    except Exception as e:
        results.append({"name": "SAM+Triton", "mean_ms": -1, "std_ms": 0, "error": str(e)})
    return results


def bench_esrgan_with_triton(runs: int, warmup: int, device: str) -> list[dict]:
    """ESRGAN end-to-end: baseline vs Triton kernel injection."""
    results = []
    model_path = ROOT / "app" / "weights" / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        results.append({"name": "ESRGAN+Triton (weights missing)", "mean_ms": -1, "std_ms": 0})
        return results

    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    try:
        import copy
        from app.core.esrgan_optimized import OptimizedRealESRGAN
        from app.core.esrgan_kernels import inject_esrgan_kernels

        # Baseline: FP16 + torch.compile (injection already happens in load())
        # Run a no-injection variant for fair comparison
        esrgan_base = OptimizedRealESRGAN(
            model_path=model_path, device=device, use_fp16=True, use_compile=True
        )
        # Temporarily disable injection in load()
        esrgan_base._skip_triton_inject = True
        if esrgan_base.load(model_path):
            mean, std = timeit(lambda: esrgan_base.upscale_image(dummy), runs, warmup)
            results.append({"name": "ESRGAN FP16+compile (baseline)", "mean_ms": mean, "std_ms": std})

            # Now inject Triton kernels on the same loaded model
            inject_esrgan_kernels(esrgan_base._upsampler.model)
            mean, std = timeit(lambda: esrgan_base.upscale_image(dummy), runs, warmup)
            results.append({"name": "ESRGAN FP16+compile+Triton", "mean_ms": mean, "std_ms": std})
            esrgan_base.unload()

    except Exception as e:
        results.append({"name": "ESRGAN+Triton", "mean_ms": -1, "std_ms": 0, "error": str(e)})
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
        # --- Custom Triton kernel micro-benchmarks (no model weights needed) ---
        "Triton LayerNorm (SAM hidden dims)": bench_triton_layernorm(args.runs, args.warmup, args.device),
        "Triton Window Ops (SAM 64×64 map)": bench_triton_window_ops(args.runs, args.warmup, args.device),
        "Triton ESRGAN Ops (per-layer)": bench_triton_esrgan_ops(args.runs, args.warmup, args.device),
        # --- End-to-end with Triton injection ---
        "SAM end-to-end: baseline vs +Triton": bench_sam_with_triton(args.runs, args.warmup, args.device),
        "ESRGAN end-to-end: baseline vs +Triton": bench_esrgan_with_triton(args.runs, args.warmup, args.device),
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
