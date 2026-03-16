# CorbeauSplat — GPU Inference Improvement Plan
*Branch: `windows-vr` | Target: Windows 11 / CUDA 12.9 / RTX 4090*

---

## 1. Current State of GFPGAN and RealESRGAN

### RealESRGAN (fully wired, partially optimised)
RealESRGAN (`realesrgan==0.3.0` / `basicsr==1.4.2`) is the **pre-COLMAP upscaler**.
Pipeline position: images extracted → **RealESRGAN upscale** → COLMAP feature extraction.

| File | Role |
|------|------|
| `app/core/upscale_engine.py` | Loads `RealESRGANer`, runs `enhance()` per image |
| `app/core/esrgan_optimized.py` | Drop-in replacement: FP16 autocast + `torch.compile(max-autotune)` + double-buffered CUDA streams |
| `app/gui/tabs/upscale_tab.py` | UI: model selector, tile size, scale (×1/2/4), FP16 toggle, face-enhance checkbox |
| `app/gui/main_window.py` | Passes `upscale_config` to `ColmapWorker` → `ColmapEngine._run_upscale()` |

UI config keys: `model_name`, `tile`, `target_scale`, `face_enhance`, `fp16`.

### GFPGAN (dependency installed, implementation is a stub)
`gfpgan==1.3.8` + `facexlib==0.3.0` are installed.
The `face_enhance` checkbox exists in the UI and is forwarded through the pipeline, **but the code
path is just `pass`** (`upscale_engine.py:243`). No actual GFPGAN inference ever runs.

---

## 2. What Will Be Implemented

### 2.1 GFPGAN face enhancement — custom Triton/CUDA-graph kernel

**Source:** `D:\VisoMaster - fusion\VisoMaster-fusion-git-dev\custom_kernels\gfpgan_v1_4\gfpgan_torch.py`
VisoMaster contains a full FP16 PyTorch reimplementation of GFPGANv1.4 (512 × 512) with three
performance tiers:

| Tier | Backend | Latency (RTX 4090) | vs ORT CUDA EP |
|------|---------|-------------------|----------------|
| Tier 1 | ORT CUDA EP (baseline) | ~ref | 1.00× |
| Tier 2 | FP16 + Triton demod + Triton fused-act | ~1.59× faster | 1.59× |
| Tier 3 | Tier 2 + CUDA graph | ~1.88× faster | **1.88×** |

Key custom ops used:
- **`triton_demod`** — fuses 7 ONNX nodes (style-modulated weight demodulation) into 2 memory passes per output channel via warp-shuffle reduction.
- **`triton_fused_gfpgan_act`** — fuses `cat([conv_bias, noise]) + LeakyReLU + scale` into one kernel (two variants: `_with_noise` / `_no_noise`).
- **CUDA graph capture** — static `(1, 3, 512, 512)` shape recorded once, then replayed every frame without Python overhead.

**Plan: create `app/core/gfpgan_engine.py`**

```
class GFPGANEngine:
    """
    GFPGAN v1.4 face restoration using the VisoMaster custom kernel path.
    Tier selection at load time: CUDA graph > Triton > CUDA C++ > PyTorch.
    """

    def load(checkpoint: Path, model_type="v1.4", device="cuda") -> bool
    def enhance_frame(bgr: np.ndarray) -> np.ndarray        # single BGR in → BGR out
    def enhance_batch(bgr_list: list[np.ndarray]) -> list   # loop over frames
    def unload() -> None
```

Implementation steps:
1. Copy `gfpgan_torch.py` from VisoMaster into `app/core/vendor/gfpgan_torch.py` (read-only vendor copy, no modifications).
2. Copy `triton_ops.py` (only the `triton_demod` and `triton_fused_gfpgan_act` kernels) into `app/core/vendor/triton_ops.py`.
3. `GFPGANEngine.load()` instantiates `GFPGANTorch`, optionally calls `build_cuda_graph_runner()`.
4. Replace the `pass` stub in `upscale_engine.py:upscale_image()` with a call to `GFPGANEngine.enhance_frame()`.
5. Add `GFPGANEngine` to `VR180Engine.process_video()` as an optional post-SAM face-restoration step.

Model weight source: `GFPGANv1.4.pth` — auto-download from
`https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth`
with SHA-256 checksum verification.

---

### 2.2 RealESRGAN via ONNX Runtime TensorRT EP

Current `esrgan_optimized.py` uses `RealESRGANer.enhance()` (PyTorch). Replace the hot path with
ONNX Runtime + `TensorrtExecutionProvider` for the highest throughput on RTX 4090.

VisoMaster's frame-enhancer pipeline uses this exact approach (ORT TRT EP is its **default** for
super-resolution) and benchmarks it as the fastest tier for frame-by-frame inference.

**Plan: extend `app/core/esrgan_optimized.py`**

```
class OptimizedRealESRGAN:

    # New methods
    def export_onnx(output_path: Path, opset: int = 17) -> bool
        """Export RRDBNet to ONNX. Called once; result cached on disk."""

    def build_trt_session(onnx_path: Path) -> bool
        """
        Build ORT InferenceSession with TensorrtExecutionProvider.
        Serialises TRT engine to disk (trt_cache/) on first call.
        Subsequent loads deserialise from cache (~200 ms vs ~60 s build).
        Provider options:
          trt_fp16_enable=True
          trt_engine_cache_enable=True
          trt_engine_cache_path=str(trt_cache_dir)
          trt_max_workspace_size=4GB
        """

    def upscale_image_trt(bgr: np.ndarray, outscale: float = 4.0) -> np.ndarray
        """Run ORT TRT session; tile internally if image > tile_size."""
```

Fallback chain (auto-selected at `load()` time):
1. ORT `TensorrtExecutionProvider` — fastest (build TRT engine on first run)
2. ORT `CUDAExecutionProvider` — fast, no build step
3. PyTorch FP16 + torch.compile — current implementation
4. PyTorch FP32 — CPU / no CUDA fallback

TRT engine cache directory: `app/weights/trt_cache/realesrgan/`

**ONNX export notes:**
- `RRDBNet.forward()` is compatible with `torch.onnx.export` (no dynamic control flow).
- Dynamic axes: `{"input": {0: "batch", 2: "height", 3: "width"}}` for tiled inference.
- Opset 17 required for `torch.compile`-exported graphs.
- Tiling wrapper: split input into `tile_size × tile_size` tiles, run each through TRT session,
  reassemble with `tile_pad` overlap blending.

---

### 2.3 Additional techniques from VisoMaster applicable to CorbeauSplat

The following VisoMaster optimizations are independent of face swap and directly applicable:

#### A. cuDNN benchmark mode (global, zero-code-change)
**Source:** `custom_kernels/__init__.py`
```python
torch.backends.cudnn.benchmark = True
```
Add to `app/core/cuda_utils.py:warm_up_cuda()`.
**Benefit:** cuDNN auto-selects the fastest convolution algorithm on first run for fixed input shapes
(all our models — RealESRGAN, SAM, GFPGAN — use fixed shapes). Free speedup, no code risk.

#### B. Triton weight-demodulation kernel for GFPGAN
**Source:** `custom_kernels/triton_ops.py` → `triton_demod()`
Already covered in §2.1. Shared with GPEN, CodeFormer — can be used for any StyleGAN2-derived
model added in the future.

#### C. CUDA graph capture for SAM encoder
**Source:** `custom_kernels/gfpgan_v1_4/gfpgan_torch.py:build_cuda_graph_runner()`
SAM's ViT image encoder runs on a fixed `(1, 3, 1024, 1024)` input (after resize).
Extend `app/core/sam_optimized.py:PersistentSAMPredictor` to capture the encoder as a CUDA graph
after the first 3 warmup forward passes.
**Benefit:** Eliminates Python kernel-launch overhead (~10–15 % per frame).

#### D. XSeg face segmentation as lightweight SAM replacement for VR180
**Source:** `custom_kernels/xseg/xseg_torch.py`
XSeg is a 256 × 256 symmetric U-Net for binary face/body masks.
In VisoMaster it achieves **5.93×** speedup vs ORT CUDA EP via Triton RMSNormMax fusion and CUDA
graphs (1.95 ms per frame on RTX 4090).

**Use case in CorbeauSplat VR180 pipeline:**
When the subject is a person filmed against a green screen, XSeg can produce a coarse foreground
mask at < 2 ms/frame. Combined with GPUChromaKey (chroma-key first, XSeg refine) this replaces
the heavy SAM ViT encoder for the common single-person use case.

**Plan:** add `app/core/xseg_engine.py`:
```
class XSegEngine:
    def load(checkpoint: Path, device="cuda") -> bool
    def predict_frame(rgb: np.ndarray) -> np.ndarray   # returns H×W uint8 mask
    def predict_batch(rgb_list) -> list[np.ndarray]
```

New VR180 param: `use_xseg: bool = False` (toggle between SAM and XSeg via UI).

#### E. Tiled RealESRGAN with CUDA stream double-buffering
**Source:** VisoMaster frame-enhancer tiling strategy (256 px tiles, overlap reassembly)
Current `esrgan_optimized.py:upscale_folder()` processes images sequentially.
Implement proper tile extraction → stream A (H2D next tile) + stream B (inference current tile)
pipelining, matching VisoMaster's approach.
**Benefit:** ~20–30 % throughput improvement on large (4K+) images.

#### F. Channels-last (NHWC) memory layout for RealESRGAN
```python
model = model.to(memory_format=torch.channels_last)
x = x.to(memory_format=torch.channels_last)
```
NCHW → NHWC gives cuDNN a fast-path for ResBlock convolutions (same as CodeFormer/RestoreFormer
in VisoMaster).
**Benefit:** ~5–10 % reduction in convolution time for standard ResBlocks.

#### G. ONNX export + ORT CUDA EP for SAM image encoder
The SAM ViT encoder accounts for ~80 % of per-frame inference time.
Exporting only the encoder to ONNX and running it via ORT `CUDAExecutionProvider` (or TRT EP) can
match or beat `torch.compile` while also eliminating JIT recompilation on first batch.

**Plan:** add `app/core/sam_optimized.py:SAMONNXEncoder` class:
```
class SAMONNXEncoder:
    def __init__(self, onnx_path: Path, use_trt: bool = True)
    def encode(rgb: np.ndarray) -> np.ndarray   # returns (256, 64, 64) features
```
`PersistentSAMPredictor.encode_batch()` delegates to `SAMONNXEncoder` when available.

---

## 3. Implementation Order

| # | Task | File(s) | Estimated speedup |
|---|------|---------|------------------|
| 1 | cuDNN benchmark=True in warm_up_cuda() | `cuda_utils.py` | ~5–15 % (free) |
| 2 | Channels-last for RealESRGAN | `esrgan_optimized.py` | ~5–10 % |
| 3 | CUDA graph capture for SAM encoder | `sam_optimized.py` | ~10–15 % |
| 4 | GFPGAN custom kernel (implement face_enhance) | `gfpgan_engine.py` (new) + `upscale_engine.py` | 1.88× vs baseline GFPGAN |
| 5 | RealESRGAN ONNX export + ORT TRT EP | `esrgan_optimized.py` | 2–3× vs torch.compile |
| 6 | Tiled RealESRGAN CUDA stream pipelining | `esrgan_optimized.py` | ~20–30 % on 4K+ |
| 7 | XSeg lightweight segmentation | `xseg_engine.py` (new) + `vr180_engine.py` | 5.93× vs SAM on RTX 4090 |
| 8 | SAM ONNX encoder via ORT TRT EP | `sam_optimized.py` | ~2× encoder speedup |

---

## 4. File Layout After Implementation

```
app/
  core/
    cuda_utils.py           ← add cuDNN benchmark=True
    esrgan_optimized.py     ← add ONNX export, ORT TRT EP, channels-last, tile pipelining
    gfpgan_engine.py        ← NEW: GFPGAN inference via VisoMaster custom kernels
    xseg_engine.py          ← NEW: XSeg fast face/body segmentation
    sam_optimized.py        ← add CUDA graph for encoder, SAMONNXEncoder class
    vendor/
      gfpgan_torch.py       ← vendor copy from VisoMaster (read-only)
      triton_ops.py         ← vendor copy: triton_demod + triton_fused_gfpgan_act only
  gui/
    tabs/
      vr180_tab.py          ← add use_xseg toggle
  weights/
    GFPGANv1.4.pth          ← auto-downloaded
    trt_cache/
      realesrgan/           ← TRT engine cache
      sam_encoder/          ← TRT engine cache
```

---

## 5. Dependency Changes Required

Add to `setup_dependencies.py` UpscaleEngineDep and VR180EngineDep install methods:

```
onnxruntime-gpu==1.24.3   # already present
tensorrt                  # for TRT engine build (Windows: pip install tensorrt)
torch-tensorrt            # optional, for torch.compile → TRT path
```

No new pip packages required for the Triton kernel path — `triton-windows==3.6.0.post26` already covers it.

---

## 6. Quality Improvements (non-speed)

Beyond speed, the following quality improvements are enabled by this plan:

| Improvement | Mechanism | Where applied |
|-------------|-----------|---------------|
| Face detail restoration | GFPGAN v1.4 on faces before COLMAP | All modes with faces |
| Higher-fidelity upscale | RealESRGAN via TRT FP16 (lower quantisation error) | Pre-COLMAP upscale |
| Cleaner segmentation masks | XSeg + GPUChromaKey combined masks | VR180 pipeline |
| Consistent COLMAP features | Higher-res, sharper face images fed to SIFT | Feature extraction |
| More Gaussians on faces | Better per-pixel detail → more COLMAP points on face regions | Gaussian splatting quality |
