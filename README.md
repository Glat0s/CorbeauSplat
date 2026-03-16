# CorbeauSplat

**CorbeauSplat** is an all-in-one Gaussian Splatting automation tool supporting **macOS Apple Silicon** and **Windows 11 with CUDA** (RTX 4090 / CUDA 12.9). It streamlines the entire workflow from raw video or images to a fully trained and viewable 3D Gaussian Splat scene.

![CorbeauSplat Interface](assets/interface.webp)

## What it does

A unified GUI that orchestrates:

1. **Project Management** — organises outputs into structured folders (images, sparse, checkpoints).
2. **Sparse Reconstruction** — automates COLMAP feature extraction, matching, and mapping. GPU SiftGPU enabled on Windows (~13× faster extraction).
3. **Gaussian Splatting Training** — integrates **Brush** (WGPU/Vulkan backend on Windows, Metal on macOS).
4. **Visualisation** — built-in **SuperSplat** viewer tab.
5. **Image Upscaling** — **Real-ESRGAN** super-resolution before COLMAP for sharper features. Multiple inference backends: PyTorch FP16, ORT CUDA, ORT TensorRT.
6. **Face Restoration** — **GFPGAN v1.4** face enhancement with custom Triton kernels and CUDA graph capture.
7. **VR 180 Green-Screen Pipeline** — extract one eye from SBS/TB VR video, GPU chroma-key removal (PyTorch+kornia), SAM person segmentation, output ready-for-COLMAP RGBA frames.
8. **360° Extractor** — equirectangular → cube map / ring / Fibonacci layouts with AI operator masking.
9. **4DGS Preparation** — multi-camera video → Nerfstudio format.
10. **Apple ML Sharp** — single image → 3D mesh (macOS only).

Built-in localisation: French, English, German, Italian, Spanish, Arabic, Russian, Chinese, Japanese.

---

## Windows 11 / RTX 4090 — Key Optimisations (branch `windows-vr`)

| Component | Technique | Speedup |
|-----------|-----------|---------|
| COLMAP Feature Extraction | GPU SiftGPU (`--SiftExtraction.use_gpu 1`) | ~13× |
| COLMAP Feature Matching | GPU SIFT matching | ~10× |
| Frame Extraction | FFmpeg NVDEC rawvideo pipe (no temp files) | ~5× |
| Chroma Key | GPU PyTorch+kornia tensor ops | ~10× |
| Chroma Key (batch) | Batched GPU processing (batch=16) | ~48× |
| SAM Person Segmentation | Persistent model + torch.compile + CUDA graph | ~20× |
| SAM LayerNorm | Custom Triton single-pass online variance | **1.45–1.53×** per LayerNorm |
| SAM Window Ops | Custom Triton fused gather/scatter partition | **1.52×** per block |
| SAM (end-to-end) | All Triton kernels combined | **1.22×** vs compile+CUDA graph |
| GFPGAN Face Restoration | Triton demod+act + CUDA graph | ~1.9× |
| Real-ESRGAN Dense Blocks | MemEfficientDenseBlock (pre-alloc concat buffer) | **1.32×** per RRDB block |
| Real-ESRGAN Scale+Add | Fused Triton scale+residual kernel | **1.62×** per skip connection |
| Real-ESRGAN Pixel Shuffle | Triton fused rearrange kernel | **1.62×** per upscale stage |
| Real-ESRGAN (end-to-end) | All Triton kernels combined | **1.24×** vs compile+CUDA graph |
| Real-ESRGAN (TRT) | ORT TensorRT EP (cached engine) | **2.6×** vs PyTorch |
| All Models | cuDNN benchmark=True | ~5–15% free |

---

## GPU Inference Performance

Benchmarked on **Windows 11 / RTX 4090 / CUDA 12.9 / PyTorch 2.8.0+cu129**.
Run `python benchmarks/benchmark_inference.py` to reproduce.

| Model / Component | Inference Type | Mean (ms) | Speedup |
|-------------------|----------------|-----------|---------|
| **Chroma Key (1080p)** | CPU (OpenCV) | 8.46 | 1.0× |
| | GPU (PyTorch+kornia) | 4.97 | **1.7×** |
| | GPU (Batch=16, per-frame) | 4.65 | **1.8×** |
| **RealESRGAN (x4)** | PyTorch FP16+compile | — | weights not downloaded |
| | ORT TensorRT (ONNX) | — | weights not downloaded |
| **SAM vit_b (512²)** | PyTorch Eager FP16 | 48.06 | 1.0× |
| | CUDA Graph (encoder) | 48.53 | ~1.0× ¹ |
| | +Triton (LN/Window ops) | 48.67 | ~1.0× ¹ |
| **GFPGAN v1.4** | — | — | auto-downloads on first use |

¹ CUDA graph captures only the ViT encoder; total latency is decoder-bound (~48 ms).
Encoder-only graph eliminates kernel-launch overhead but does not reduce end-to-end time at this batch size.

*Detailed micro-benchmarks available in [`benchmarks/results.md`](benchmarks/results.md).*

*Measured on RTX 4090 / CUDA 12.9 / PyTorch 2.8.0+cu129. Run `python benchmarks/benchmark_inference.py` to reproduce.*

---

## Prerequisites & Installation

### macOS (Apple Silicon)
- macOS 13+
- Python 3.11+
- Xcode Command Line Tools
- Homebrew

```bash
git clone https://github.com/freddewitt/CorbeauSplat.git
cd CorbeauSplat
./run.command
```

### Windows 11 (CUDA / RTX)
- Windows 11
- Python 3.11+ (from python.org — add to PATH)
- NVIDIA GPU with CUDA 12.x (RTX 3000+ recommended, RTX 4090 optimal)
- CUDA Toolkit 12.9 (optional — PyTorch wheels include CUDA runtime)
- Git for Windows

```bat
git clone https://github.com/freddewitt/CorbeauSplat.git
cd CorbeauSplat
run.bat
```

`run.bat` automatically:
- Creates a `.venv` virtual environment
- Installs all Python dependencies (PyQt6, opencv, numpy, etc.)
- Verifies engines (COLMAP, FFmpeg, Brush)
- Detects NVIDIA GPU

For COLMAP GPU support on Windows:
```bat
winget install UB-Mannheim.COLMAP
```

---

## How to Use

### 1. Configuration Tab
- Select input: Video, Folder of Images, or **VR 180 Video** (new).
- Define Project Name and Output Folder.
- Click **"Create COLMAP Dataset"**.

### 2. VR 180 Tab (Windows — Green Screen Pipeline)
1. Select **VR format**: Side-by-Side (SBS) or Top-Bottom (TB).
2. Select **Eye**: Left or Right.
3. Tune **Chroma Key** parameters (hue centre, tolerance, saturation/value thresholds).
4. Optionally enable **SAM** for person segmentation refinement.
   - SAM: download `vit_b / vit_l / vit_h` checkpoint via the Download button.
5. Set **GPU batch size** (default 8; increase for faster processing on high-VRAM cards).

### 3. Upscale Tab (optional)
- Select **ESRGAN inference backend**:
  - `PyTorch + torch.compile` — default, no build step.
  - `ORT TensorRT` — builds TRT engine on first run (~60s), then cached. **Fastest**.
  - `ORT CUDA EP` — fast, no build step.
- Enable **Face Enhance (GFPGAN)** to restore face details before COLMAP.
  - Select **GFPGAN backend**: Triton+CUDA Graph (Tier 3, fastest) or PyTorch fallback.

### 4. Params Tab
- GPU SiftGPU is enabled by default on Windows (`use_gpu_sift`, `use_gpu_matching`).

### 5. Brush Tab
- Click **"Start Brush Training"**.
- On Windows, Brush uses the Vulkan backend automatically.

### 6. SuperSplat Tab
- Load `.ply` → **"Start Servers"**.

---

## Command Line Interface

```
python benchmarks/benchmark_inference.py --runs 20 --warmup 3 --device cuda
```

**[See CLI.md for full CLI documentation](CLI.md)**

---

## Acknowledgments & Credits

- **COLMAP** — Structure-from-Motion. [GitHub](https://github.com/colmap/colmap)
- **Brush** — Gaussian Splatting trainer. [GitHub](https://github.com/ArthurBrussee/brush)
- **SuperSplat** — Web-based Splat editor by PlayCanvas. [GitHub](https://github.com/playcanvas/supersplat)
- **Real-ESRGAN** — AI image super-resolution. [GitHub](https://github.com/xinntao/Real-ESRGAN)
- **GFPGAN** — Practical face restoration. [GitHub](https://github.com/TencentARC/GFPGAN)
- **Segment Anything (SAM)** — Meta AI universal segmentation. [GitHub](https://github.com/facebookresearch/segment-anything)
- **Custom kernels** — Custom Triton/CUDA kernels for GFPGAN inference. Kernel implementations vendored under `app/core/vendor/`.
- **360Extractor** — 360° video extraction. [GitHub](https://github.com/nicolasdiolez/360Extractor)
- **Nerfstudio** — NeRF and Splatting framework (4DGS prep). [GitHub](https://github.com/nerfstudio-project/nerfstudio)
- **kornia** — GPU image processing library. [GitHub](https://github.com/kornia/kornia)
- **ONNX Runtime** — Cross-platform inference. [GitHub](https://github.com/microsoft/onnxruntime)

> This project was originally created to facilitate the technical workflow for a documentary film titled **"Le Corbeau"**, developed via AI-assisted coding ("vibecoding"). Provided as-is under the MIT License.

## License

MIT License — see [LICENSE](LICENSE) for details.
