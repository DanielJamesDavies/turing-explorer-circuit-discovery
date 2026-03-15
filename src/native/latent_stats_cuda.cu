#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel 1: fused_scatter_kernel
//
// Each thread processes one element of the flat [N] top_acts / top_indices
// arrays. Skips zero (inactive) values. For each active value, atomically
// accumulates all five statistics into output arrays in one pass:
//   n_b       += 1
//   sum_b     += v
//   sum_abs_b += |v|
//   sum_sq_b  += v^2
//   sum_sq_abs_b += |v|^2
// ---------------------------------------------------------------------------
__global__ void fused_scatter_kernel(
    const float*   acts,          // [N] flattened top_acts
    const int32_t* indices,       // [N] flattened top_indices
    float*         n_b,           // [d_sae]
    float*         sum_b,         // [d_sae]
    float*         sum_abs_b,     // [d_sae]
    float*         sum_sq_b,      // [d_sae]
    float*         sum_sq_abs_b,  // [d_sae]
    int64_t        N
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float v = acts[i];
    if (v <= 0.0f) return;

    int32_t idx = indices[i];
    float v_abs     = fabsf(v);
    float v_sq      = v * v;
    float v_abs_sq  = v_abs * v_abs;

    atomicAdd(&n_b[idx],          1.0f);
    atomicAdd(&sum_b[idx],        v);
    atomicAdd(&sum_abs_b[idx],    v_abs);
    atomicAdd(&sum_sq_b[idx],     v_sq);
    atomicAdd(&sum_sq_abs_b[idx], v_abs_sq);
}

// ---------------------------------------------------------------------------
// Kernel 2: welford_merge_kernel
//
// One thread per latent index d in [0, d_sae).
// No atomics needed — each thread owns exactly one latent slot.
//
// Computes batch mean and M2 from the scatter outputs, then applies the
// parallel Welford merge formula to update the global running statistics
// and activation count in-place.
// ---------------------------------------------------------------------------
__global__ void welford_merge_kernel(
    const float*   n_b,           // [d_sae] batch counts (float for arithmetic)
    const float*   sum_b,         // [d_sae]
    const float*   sum_abs_b,     // [d_sae]
    const float*   sum_sq_b,      // [d_sae]
    const float*   sum_sq_abs_b,  // [d_sae]
    float*         mean,          // [d_sae] running mean      — updated in-place
    float*         m2,            // [d_sae] running M2        — updated in-place
    float*         mean_abs,      // [d_sae] running mean_abs  — updated in-place
    float*         m2_abs,        // [d_sae] running M2_abs    — updated in-place
    int64_t*       n_global,      // [d_sae] running count     — updated in-place
    int64_t        d_sae
) {
    int64_t d = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_sae) return;

    float nb = n_b[d];
    if (nb == 0.0f) return;

    // Batch statistics
    float sb      = sum_b[d];
    float sb_abs  = sum_abs_b[d];
    float ssq     = sum_sq_b[d];
    float ssq_abs = sum_sq_abs_b[d];

    float mean_b     = sb     / nb;
    float mean_abs_b = sb_abs / nb;
    float m2_b       = ssq     - nb * mean_b     * mean_b;
    float m2_abs_b   = ssq_abs - nb * mean_abs_b * mean_abs_b;

    // Parallel Welford merge
    float na      = (float)n_global[d];
    float n_total = na + nb;
    float safe_n  = fmaxf(n_total, 1.0f);

    // mean
    float delta    = mean_b - mean[d];
    mean[d]       += delta * (nb / safe_n);
    m2[d]         += m2_b + delta * delta * (na * nb / safe_n);

    // mean_abs
    float delta_abs = mean_abs_b - mean_abs[d];
    mean_abs[d]    += delta_abs * (nb / safe_n);
    m2_abs[d]      += m2_abs_b + delta_abs * delta_abs * (na * nb / safe_n);

    // count
    n_global[d] += (int64_t)nb;
}

// ---------------------------------------------------------------------------
// C++ wrapper: update_latent_stats
//
// Accepts the raw [B, S, K] top_acts / top_indices tensors directly (no
// Python-side mask extraction needed), plus the state tensors to update.
// Allocates temporary scatter buffers using torch::zeros (fast, device-local).
// ---------------------------------------------------------------------------
void update_latent_stats(
    torch::Tensor top_acts,    // [B, S, K] float32 CUDA
    torch::Tensor top_indices, // [B, S, K] int32  CUDA
    torch::Tensor mean,        // [d_sae] float32 CUDA — in-place
    torch::Tensor m2,          // [d_sae] float32 CUDA — in-place
    torch::Tensor mean_abs,    // [d_sae] float32 CUDA — in-place
    torch::Tensor m2_abs,      // [d_sae] float32 CUDA — in-place
    torch::Tensor n_global,    // [d_sae] int64  CUDA  — in-place
    int64_t       d_sae
) {
    TORCH_CHECK(top_acts.dtype()    == torch::kFloat32, "top_acts must be float32");
    TORCH_CHECK(top_indices.dtype() == torch::kInt32,   "top_indices must be int32");
    TORCH_CHECK(top_acts.is_cuda(),    "top_acts must be a CUDA tensor");
    TORCH_CHECK(top_indices.is_cuda(), "top_indices must be a CUDA tensor");
    TORCH_CHECK(top_acts.is_contiguous(),    "top_acts must be contiguous");
    TORCH_CHECK(top_indices.is_contiguous(), "top_indices must be contiguous");

    int64_t N = top_acts.numel();
    if (N == 0) return;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(top_acts.device());

    // Temporary scatter accumulators — zeroed before each call
    torch::Tensor nb_t          = torch::zeros({d_sae}, opts);
    torch::Tensor sum_b_t       = torch::zeros({d_sae}, opts);
    torch::Tensor sum_abs_b_t   = torch::zeros({d_sae}, opts);
    torch::Tensor sum_sq_b_t    = torch::zeros({d_sae}, opts);
    torch::Tensor sum_sq_abs_t  = torch::zeros({d_sae}, opts);

    float*   acts_ptr    = top_acts.data_ptr<float>();
    int32_t* indices_ptr = top_indices.data_ptr<int32_t>();
    float*   nb_ptr      = nb_t.data_ptr<float>();
    float*   sum_ptr     = sum_b_t.data_ptr<float>();
    float*   sum_abs_ptr = sum_abs_b_t.data_ptr<float>();
    float*   sum_sq_ptr  = sum_sq_b_t.data_ptr<float>();
    float*   sum_sqabs_ptr = sum_sq_abs_t.data_ptr<float>();

    // Kernel 1: scatter
    constexpr int BLOCK = 256;
    int grid_scatter = (int)((N + BLOCK - 1) / BLOCK);
    fused_scatter_kernel<<<grid_scatter, BLOCK>>>(
        acts_ptr, indices_ptr,
        nb_ptr, sum_ptr, sum_abs_ptr, sum_sq_ptr, sum_sqabs_ptr,
        N
    );

    // Kernel 2: Welford merge
    int grid_welford = (int)((d_sae + BLOCK - 1) / BLOCK);
    welford_merge_kernel<<<grid_welford, BLOCK>>>(
        nb_ptr, sum_ptr, sum_abs_ptr, sum_sq_ptr, sum_sqabs_ptr,
        mean.data_ptr<float>(),
        m2.data_ptr<float>(),
        mean_abs.data_ptr<float>(),
        m2_abs.data_ptr<float>(),
        n_global.data_ptr<int64_t>(),
        d_sae
    );
}

bool has_cuda() { return true; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("has_cuda", &has_cuda, "Returns true — this build includes CUDA support");
    m.def("update_latent_stats", &update_latent_stats,
          "Fused scatter + Welford merge for latent activation statistics.\n\n"
          "Updates mean, m2, mean_abs, m2_abs, and n_global in-place.\n\n"
          "Args:\n"
          "  top_acts:    [B, S, K] float32 CUDA — SAE top-k activations\n"
          "  top_indices: [B, S, K] int32  CUDA — SAE top-k latent indices\n"
          "  mean:        [d_sae] float32 CUDA — running mean (in-place)\n"
          "  m2:          [d_sae] float32 CUDA — running M2 (in-place)\n"
          "  mean_abs:    [d_sae] float32 CUDA — running mean of |a| (in-place)\n"
          "  m2_abs:      [d_sae] float32 CUDA — running M2 of |a| (in-place)\n"
          "  n_global:    [d_sae] int64  CUDA — activation count (in-place)\n"
          "  d_sae:       int — SAE dictionary size\n",
          py::arg("top_acts"),
          py::arg("top_indices"),
          py::arg("mean"),
          py::arg("m2"),
          py::arg("mean_abs"),
          py::arg("m2_abs"),
          py::arg("n_global"),
          py::arg("d_sae")
    );
}
