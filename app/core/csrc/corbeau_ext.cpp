/*
 * corbeau_ext.cpp — pybind11 bindings for corbeau_kernels.cu
 *
 * Uses <ATen/ATen.h> instead of <torch/extension.h> to avoid MSVC 19.37
 * incompatibilities in torch/nn/cloneable.h.
 */
// torch/csrc/utils/pybind.h provides at::Tensor<->Python type_caster and
// pybind11 without pulling in torch/nn (avoids MSVC 19.37 cloneable.h bug).
#include <torch/csrc/utils/pybind.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

// Forward declarations of launchers from corbeau_kernels.cu
extern "C" {
void launch_layer_norm(const void*, void*, const void*, const void*,
                       int, int, float, int, cudaStream_t);
void launch_window_partition(const void*, void*, int, int, int, int, int, cudaStream_t);
void launch_window_unpartition(const void*, void*, int, int, int, int, int, cudaStream_t);
void launch_leaky_relu_scale_add(const void*, const void*, void*, float, float, int, cudaStream_t);
void launch_pixel_shuffle_2x(const void*, void*, int, int, int, int, cudaStream_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// layer_norm_fwd
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor layer_norm_fwd(
    at::Tensor x,      // (M, N) fp16
    at::Tensor weight, // (N,)   fp16
    at::Tensor bias,   // (N,)   fp16
    float eps,
    bool fuse_gelu)
{
    TORCH_CHECK(x.is_cuda(),    "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == at::kHalf, "x must be fp16");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const int M = (int)(x.numel() / x.size(-1));
    const int N = (int)x.size(-1);

    auto y = at::empty_like(x);

    at::cuda::CUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_layer_norm(
        x.data_ptr(), y.data_ptr(),
        weight.data_ptr(), bias.data_ptr(),
        M, N, eps, (int)fuse_gelu, stream);

    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// window_partition_fwd  (B, H, W, C) → (B*nH*nW, win, win, C)
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor window_partition_fwd(
    at::Tensor x,   // (B, H, W, C) fp16, C % 8 == 0
    int window_size)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous());
    TORCH_CHECK(x.dtype() == at::kHalf);
    const int B = (int)x.size(0), H = (int)x.size(1);
    const int W = (int)x.size(2), C = (int)x.size(3);
    TORCH_CHECK(C % 8 == 0,  "C must be divisible by 8 for float4 kernel");
    TORCH_CHECK(H % window_size == 0 && W % window_size == 0);

    const int nH = H / window_size, nW = W / window_size;
    auto out = at::empty({(long long)B * nH * nW, window_size, window_size, C},
                         x.options());

    at::cuda::CUDAGuard guard(x.device());
    launch_window_partition(
        x.data_ptr(), out.data_ptr(),
        B, H, W, C, window_size,
        at::cuda::getCurrentCUDAStream());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// window_unpartition_fwd  (B*nH*nW, win, win, C) → (B, H, W, C)
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor window_unpartition_fwd(
    at::Tensor wins,   // (B*nH*nW, win, win, C)
    int window_size, int H, int W)
{
    TORCH_CHECK(wins.is_cuda() && wins.is_contiguous());
    TORCH_CHECK(wins.dtype() == at::kHalf);
    const int C = (int)wins.size(3);
    const int nH = H / window_size, nW = W / window_size;
    const int B  = (int)(wins.size(0) / (nH * nW));

    auto out = at::empty({B, H, W, C}, wins.options());

    at::cuda::CUDAGuard guard(wins.device());
    launch_window_unpartition(
        wins.data_ptr(), out.data_ptr(),
        B, H, W, C, window_size,
        at::cuda::getCurrentCUDAStream());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// leaky_relu_scale_add  out = LeakyReLU(x, slope) * scale + residual
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor leaky_relu_scale_add(
    at::Tensor x,        // fp16, any shape, n % 2 == 0
    at::Tensor residual,
    float neg_slope,
    float scale)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kHalf);
    TORCH_CHECK(x.numel() % 2 == 0);
    auto out = at::empty_like(x);

    at::cuda::CUDAGuard guard(x.device());
    launch_leaky_relu_scale_add(
        x.contiguous().data_ptr(),
        residual.contiguous().data_ptr(),
        out.data_ptr(),
        neg_slope, scale, (int)x.numel(),
        at::cuda::getCurrentCUDAStream());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// pixel_shuffle_2x  (B, 4C, H, W) → (B, C, 2H, 2W)
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor pixel_shuffle_2x(at::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == at::kHalf);
    TORCH_CHECK(x.dim() == 4 && x.size(1) % 4 == 0);
    const int B = (int)x.size(0), C4 = (int)x.size(1);
    const int H = (int)x.size(2), W  = (int)x.size(3);
    const int C = C4 / 4;
    auto out = at::empty({B, C, H*2, W*2}, x.options());

    at::cuda::CUDAGuard guard(x.device());
    launch_pixel_shuffle_2x(
        x.contiguous().data_ptr(), out.data_ptr(),
        B, C, H, W,
        at::cuda::getCurrentCUDAStream());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CorbeauSplat custom CUDA kernels";
    m.def("layer_norm_fwd",          &layer_norm_fwd,
          "Single-pass warp-shuffle LayerNorm (+ optional tanh-GELU)",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("eps")=1e-6f, py::arg("fuse_gelu")=false);
    m.def("window_partition_fwd",    &window_partition_fwd,
          "float4-vectorised SAM window partition",
          py::arg("x"), py::arg("window_size"));
    m.def("window_unpartition_fwd",  &window_unpartition_fwd,
          "float4-vectorised SAM window unpartition",
          py::arg("wins"), py::arg("window_size"), py::arg("H"), py::arg("W"));
    m.def("leaky_relu_scale_add",    &leaky_relu_scale_add,
          "Fused LeakyReLU*scale + residual (ESRGAN RRDB skip)",
          py::arg("x"), py::arg("residual"),
          py::arg("neg_slope")=0.2f, py::arg("scale")=0.2f);
    m.def("pixel_shuffle_2x",        &pixel_shuffle_2x,
          "Coalesced pixel shuffle 2x (ESRGAN upsampling)");
}
