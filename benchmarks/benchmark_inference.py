#!/usr/bin/env python3
"""
CorbeauSplat GPU Inference Benchmarks
======================================
Tests all inference backends on Windows 11 / RTX 4090 (CUDA 12.4 / torch 2.6.0).

Usage:
    cd D:/CorbeauSplat
    .venv/Scripts/python.exe benchmarks/benchmark_inference.py [--runs N] [--warmup N]

Output: Markdown table printed to stdout + saved to benchmarks/results.md
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def timeit(fn: Callable, runs: int = 20, warmup: int = 3) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) after warmup un-timed calls."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def bench_chroma_key(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    try:
        from app.core.gpu_chroma_key import GPUChromaKey
        import cv2, numpy as np
        dummy = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        ck_cpu = GPUChromaKey(device="cpu")
        mean, std = timeit(lambda: ck_cpu.process_frame(dummy), runs, warmup)
        results.append({"name": "ChromaKey CPU (OpenCV)", "mean_ms": mean, "std_ms": std})

        if device == "cuda":
            ck_gpu = GPUChromaKey(device="cuda")
            mean, std = timeit(lambda: ck_gpu.process_frame(dummy), runs, warmup)
            results.append({"name": "ChromaKey GPU (PyTorch+kornia)", "mean_ms": mean, "std_ms": std})

            batch = [dummy] * 16
            mean, std = timeit(lambda: ck_gpu.process_batch(batch), runs, warmup)
            results.append({"name": "ChromaKey GPU batch=16 (per-frame)", "mean_ms": mean / 16, "std_ms": std / 16})
    except Exception as e:
        results.append({"name": "ChromaKey", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


def bench_esrgan(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    weights_dir = ROOT / "app" / "weights"
    model_path  = weights_dir / "RealESRGAN_x4plus.pth"
    if not model_path.exists():
        results.append({"name": "ESRGAN (weights missing)", "mean_ms": -1, "std_ms": 0, "notes": "download first"})
        return results
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    try:
        from app.core.esrgan_optimized import OptimizedRealESRGAN
        from app.core.esrgan_kernels import inject_esrgan_kernels

        # Baseline: FP16 + torch.compile
        esrgan_base = OptimizedRealESRGAN(model_path=model_path, device=device, use_fp16=True, use_compile=True)
        if esrgan_base.load(model_path):
            mean, std = timeit(lambda: esrgan_base.upscale_image(dummy), runs, warmup)
            results.append({"name": "ESRGAN FP16+compile", "mean_ms": mean, "std_ms": std})

            # +Triton/CUDA kernels already injected on load; benchmark again explicitly
            inject_esrgan_kernels(esrgan_base._upsampler.model)
            mean, std = timeit(lambda: esrgan_base.upscale_image(dummy), runs, warmup)
            results.append({"name": "ESRGAN FP16+compile+CUDA kernels", "mean_ms": mean, "std_ms": std})
            esrgan_base.unload()

        # ORT path
        try:
            import onnxruntime as ort
            esrgan_ort = OptimizedRealESRGAN(model_path=model_path, device=device, use_fp16=False, use_compile=False)
            if esrgan_ort.load(model_path):
                onnx_path = weights_dir / "realesrgan.onnx"
                if not onnx_path.exists():
                    esrgan_ort.export_onnx(onnx_path)
                if onnx_path.exists() and esrgan_ort.build_trt_session(onnx_path):
                    mean, std = timeit(lambda: esrgan_ort.upscale_image_trt(dummy), runs, warmup)
                    results.append({"name": "ESRGAN ORT TRT EP", "mean_ms": mean, "std_ms": std})
                esrgan_ort.unload()
        except Exception as e:
            results.append({"name": "ESRGAN ORT TRT EP", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    except Exception as e:
        results.append({"name": "ESRGAN", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


def bench_sam(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    ckpt = ROOT / "engines" / "sam_vit_b_01ec64.pth"
    if not ckpt.exists():
        results.append({"name": "SAM (checkpoint missing)", "mean_ms": -1, "std_ms": 0, "notes": "download first"})
        return results
    dummy_rgb = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        from app.core.sam_optimized import PersistentSAMPredictor
        from app.core.sam_kernels import inject_sam_triton_kernels

        sam_base = PersistentSAMPredictor(str(ckpt), "vit_b", device=device, use_fp16=True, use_compile=False)
        if sam_base.is_loaded:
            mean, std = timeit(lambda: sam_base.predict_frame(dummy_rgb), runs // 2, warmup)
            results.append({"name": "SAM vit_b eager FP16", "mean_ms": mean, "std_ms": std})
            sam_base.unload()

        # CUDA graph + eager: use PersistentSAMPredictor with graph capture
        sam_graph = PersistentSAMPredictor(str(ckpt), "vit_b", device=device, use_fp16=True, use_compile=False)
        if sam_graph.is_loaded:
            sam_warmup = max(10, warmup)
            mean, std = timeit(lambda: sam_graph.predict_frame(dummy_rgb), runs // 2, sam_warmup)
            results.append({"name": "SAM vit_b CUDA graph (encoder)", "mean_ms": mean, "std_ms": std,
                             "notes": "CUDA graph on encoder; decoder runs eager"})

            # Inject Triton/Flash-Attention kernels on top of the CUDA graph encoder
            if hasattr(sam_graph._sam, "image_encoder"):
                try:
                    inject_sam_triton_kernels(sam_graph._sam.image_encoder)
                    mean, std = timeit(lambda: sam_graph.predict_frame(dummy_rgb), runs // 2, sam_warmup)
                    results.append({"name": "SAM vit_b +Flash+LN+WindowOps", "mean_ms": mean, "std_ms": std})
                except Exception as e:
                    results.append({"name": "SAM vit_b +Flash+LN+WindowOps", "mean_ms": -1, "std_ms": 0,
                                    "notes": str(e)})
            sam_graph.unload()
    except Exception as e:
        results.append({"name": "SAM", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


def bench_gfpgan(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    ckpt = ROOT / "app" / "weights" / "GFPGANv1.4.pth"
    if not ckpt.exists():
        results.append({"name": "GFPGAN (weights missing)", "mean_ms": -1, "std_ms": 0, "notes": "auto-downloads on use"})
        return results
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        from app.core.gfpgan_engine import GFPGANEngine
        for use_graph, label in [(False, "Tier 2"), (True, "Tier 3")]:
            g = GFPGANEngine(checkpoint=ckpt, device=device, use_fp16=True, use_cuda_graph=use_graph)
            if g.load():
                mean, std = timeit(lambda: g.enhance_frame(dummy), runs, warmup)
                results.append({"name": f"GFPGAN Triton {label}", "mean_ms": mean, "std_ms": std})
                g.unload()
    except Exception as e:
        results.append({"name": "GFPGAN", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results



# -- Micro-benchmarks (no model weights needed) --------------------------------

def bench_triton_layernorm(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    if device != "cuda":
        return results
    try:
        from app.core.vendor.sam_triton_kernels import TritonLayerNorm
        from app.core.cuda_ext import layer_norm_cuda, ext_available

        for dim in (768, 1024, 1280):
            x    = torch.randn(4096, dim, device=device, dtype=torch.float16)
            ln_pt = nn.LayerNorm(dim, eps=1e-6).to(device).half()

            mean, std = timeit(lambda: ln_pt(x), runs, warmup)
            results.append({"name": f"LayerNorm-{dim} PyTorch", "mean_ms": mean, "std_ms": std})

            ln_tri = TritonLayerNorm(dim, eps=1e-6).to(device)
            ln_tri.weight.data.copy_(ln_pt.weight.data)
            ln_tri.bias.data.copy_(ln_pt.bias.data)
            mean, std = timeit(lambda: ln_tri(x), runs, warmup)
            results.append({"name": f"LayerNorm-{dim} Triton", "mean_ms": mean, "std_ms": std})

            if ext_available():
                w = ln_pt.weight.half().contiguous()
                b = ln_pt.bias.half().contiguous()
                mean, std = timeit(lambda: layer_norm_cuda(x, w, b, 1e-6, False), runs, warmup)
                results.append({"name": f"LayerNorm-{dim} CUDA ext", "mean_ms": mean, "std_ms": std})

            ln_tri_gelu = TritonLayerNorm(dim, eps=1e-6, fuse_gelu=True).to(device)
            ln_tri_gelu.weight.data.copy_(ln_pt.weight.data)
            ln_tri_gelu.bias.data.copy_(ln_pt.bias.data)
            mean, std = timeit(lambda: ln_tri_gelu(x), runs, warmup)
            results.append({"name": f"LayerNorm+GELU-{dim} Triton fused", "mean_ms": mean, "std_ms": std})

            if ext_available():
                mean, std = timeit(lambda: layer_norm_cuda(x, w, b, 1e-6, True), runs, warmup)
                results.append({"name": f"LayerNorm+GELU-{dim} CUDA ext fused", "mean_ms": mean, "std_ms": std})

    except Exception as e:
        results.append({"name": "Triton/CUDA LayerNorm", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


def bench_triton_window_ops(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    if device != "cuda":
        return results
    try:
        from app.core.vendor.sam_triton_kernels import triton_window_partition
        from app.core.cuda_ext import window_partition_cuda, window_unpartition_cuda, ext_available

        # SAM vit_b: 1024px → 64×64 feature map padded to 70×70 for ws=14.
        # Use 56×56 = 4×14 — cleanly divisible, representative of real usage.
        B, H, W, C = 1, 56, 56, 768
        x = torch.randn(B, H, W, C, device=device, dtype=torch.float16)
        ws = 14
        nH, nW = H // ws, W // ws

        def pt_part(t):
            t2 = t.view(B, H // ws, ws, W // ws, ws, C)
            return t2.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

        mean, std = timeit(lambda: pt_part(x), runs, warmup)
        results.append({"name": "Window partition PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_window_partition(x, ws), runs, warmup)
        results.append({"name": "Window partition Triton", "mean_ms": mean, "std_ms": std})

        if ext_available():
            mean, std = timeit(lambda: window_partition_cuda(x, ws), runs, warmup)
            results.append({"name": "Window partition CUDA ext (float4)", "mean_ms": mean, "std_ms": std})

            wins, pad_hw = window_partition_cuda(x, ws)
            hw_orig = (H, W)
            mean, std = timeit(lambda: window_unpartition_cuda(wins, ws, pad_hw, hw_orig), runs, warmup)
            results.append({"name": "Window unpartition CUDA ext (float4)", "mean_ms": mean, "std_ms": std})

    except Exception as e:
        results.append({"name": "Triton/CUDA window ops", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


def bench_triton_esrgan_ops(runs: int, warmup: int, device: str) -> list[dict]:
    results = []
    if device != "cuda":
        return results
    try:
        from app.core.vendor.esrgan_triton_kernels import (
            triton_scale_add, triton_leakyrelu_inplace,
            triton_pixel_shuffle_2x, MemEfficientDenseBlock,
        )
        from app.core.cuda_ext import leaky_relu_scale_add_cuda, pixel_shuffle_2x_cuda, ext_available

        x = torch.randn(1, 64, 128, 128, device=device, dtype=torch.float16)
        r = torch.randn_like(x)

        mean, std = timeit(lambda: x * 0.2 + r, runs, warmup)
        results.append({"name": "scale_add PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_scale_add(x, r, 0.2), runs, warmup)
        results.append({"name": "scale_add Triton", "mean_ms": mean, "std_ms": std})

        if ext_available():
            mean, std = timeit(lambda: leaky_relu_scale_add_cuda(x, r, 0.2, 0.2), runs, warmup)
            results.append({"name": "LeakyReLU+scale_add CUDA ext (fused)", "mean_ms": mean, "std_ms": std})

        ps_in = torch.randn(1, 256, 256, 256, device=device, dtype=torch.float16)
        mean, std = timeit(lambda: F.pixel_shuffle(ps_in, 2), runs, warmup)
        results.append({"name": "PixelShuffle-2x PyTorch", "mean_ms": mean, "std_ms": std})

        mean, std = timeit(lambda: triton_pixel_shuffle_2x(ps_in), runs, warmup)
        results.append({"name": "PixelShuffle-2x Triton", "mean_ms": mean, "std_ms": std})

        if ext_available():
            mean, std = timeit(lambda: pixel_shuffle_2x_cuda(ps_in), runs, warmup)
            results.append({"name": "PixelShuffle-2x CUDA ext", "mean_ms": mean, "std_ms": std})

        db = MemEfficientDenseBlock(64, 32).to(device).half()
        feat = torch.randn(1, 64, 64, 64, device=device, dtype=torch.float16)
        mean, std = timeit(lambda: db(feat), runs, warmup)
        results.append({"name": "DenseBlock MemEfficient (Triton+CUDA)", "mean_ms": mean, "std_ms": std})

    except Exception as e:
        results.append({"name": "Triton/CUDA ESRGAN ops", "mean_ms": -1, "std_ms": 0, "notes": str(e)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def format_table(sections: dict) -> str:
    lines = ["| Benchmark | Mean (ms) | Std (ms) | Notes |",
             "|-----------|-----------|----------|-------|"]
    for section, rows in sections.items():
        lines.append(f"| **{section}** | | | |")
        for r in rows:
            if r["mean_ms"] < 0:
                note = r.get("notes", r.get("error", "skipped"))
                lines.append(f"| {r['name']} | - | - | {note} |")
            else:
                note = r.get("notes", "")
                lines.append(f"| {r['name']} | {r['mean_ms']:.2f} | {r['std_ms']:.2f} | {note} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="CorbeauSplat inference benchmarks")
    parser.add_argument("--runs",   type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    print("CorbeauSplat Inference Benchmarks")
    print(f"Device: {args.device} | Runs: {args.runs} | Warmup: {args.warmup}")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
        from app.core.cuda_ext import ext_available
        print(f"CUDA ext compiled: {ext_available()}")
    print()

    def _run(fn):
        """Run a benchmark section, flushing GPU state first."""
        import gc
        gc.collect()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        return fn()

    # Run micro-benchmarks BEFORE SAM to avoid CUDA graph allocator interference.
    # SAM's _build_encoder_graph allocates static tensors inside a CUDA graph capture;
    # freeing these afterwards confuses the CUDACachingAllocator if subsequent
    # allocations happen in the same process.
    sections = {
        "Chroma Key (1920x1080)":               _run(lambda: bench_chroma_key(args.runs, args.warmup, args.device)),
        "RealESRGAN (256x256 -> 1024x1024)":    _run(lambda: bench_esrgan(args.runs, args.warmup, args.device)),
        "GFPGAN (512x512 face)":                _run(lambda: bench_gfpgan(args.runs, args.warmup, args.device)),
        "Micro: LayerNorm (4096 tokens)":       _run(lambda: bench_triton_layernorm(args.runs, args.warmup, args.device)),
        "Micro: Window ops (56x56, ws=14)":     _run(lambda: bench_triton_window_ops(args.runs, args.warmup, args.device)),
        "Micro: ESRGAN ops":                    _run(lambda: bench_triton_esrgan_ops(args.runs, args.warmup, args.device)),
        # SAM last — its CUDA graph static tensors interfere with later allocations
        "SAM vit_b (512x512 frame)":            _run(lambda: bench_sam(args.runs, args.warmup, args.device)),
    }

    table = format_table(sections)
    print(table)

    out = ROOT / "benchmarks" / "results.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# CorbeauSplat Inference Benchmark Results\n\n")
        f.write(f"**Platform:** Windows 11 / RTX 4090 / CUDA 12.4 / PyTorch {torch.__version__}  \n")
        f.write(f"**Runs:** {args.runs} | **Warmup:** {args.warmup}  \n\n")
        f.write(table)
        f.write("\n\n*Generated by `benchmarks/benchmark_inference.py`*\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
