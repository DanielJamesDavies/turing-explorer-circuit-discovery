#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

static constexpr size_t WORKSPACE_SIZE = 16ULL * 1024 * 1024; // 16 MB
static constexpr int TOPK_THREADS = 256;

#define CUBLASLT_CHECK(call)                                                \
  do {                                                                      \
    cublasStatus_t _s = (call);                                             \
    TORCH_CHECK(_s == CUBLAS_STATUS_SUCCESS,                                \
                "cublasLt error ", static_cast<int>(_s), " in " #call);    \
  } while (0)

__device__ __forceinline__ bool better_pair(
    float value,
    int index,
    float best_value,
    int best_index) {
  return value > best_value || (value == best_value && (best_index < 0 || index < best_index));
}

__global__ void local_topk_kernel(
    const c10::BFloat16* __restrict__ acts,
    c10::BFloat16* __restrict__ candidate_values,
    int32_t* __restrict__ candidate_indices,
    int64_t rows,
    int64_t width,
    int64_t candidate_width,
    int64_t candidate_offset,
    int64_t global_start,
    int local_k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int* shared_indices = reinterpret_cast<int*>(shared_values + blockDim.x);
  int* selected = shared_indices + blockDim.x;

  const int tid = threadIdx.x;
  const int64_t row_base = row * width;
  const int64_t out_base = row * candidate_width + candidate_offset;

  for (int out = 0; out < local_k; ++out) {
    float best_value = -CUDART_INF_F;
    int best_index = -1;

    for (int64_t col = tid; col < width; col += blockDim.x) {
      bool already_selected = false;
      for (int prev = 0; prev < out; ++prev) {
        if (selected[prev] == static_cast<int>(col)) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }

      const float value = static_cast<float>(acts[row_base + col]);
      const int index = static_cast<int>(col);
      if (better_pair(value, index, best_value, best_index)) {
        best_value = value;
        best_index = index;
      }
    }

    shared_values[tid] = best_value;
    shared_indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = shared_values[tid + stride];
        const int other_index = shared_indices[tid + stride];
        if (better_pair(other_value, other_index, shared_values[tid], shared_indices[tid])) {
          shared_values[tid] = other_value;
          shared_indices[tid] = other_index;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected[out] = shared_indices[0];
      candidate_values[out_base + out] = c10::BFloat16(shared_values[0]);
      candidate_indices[out_base + out] = static_cast<int32_t>(global_start + shared_indices[0]);
    }
    __syncthreads();
  }
}

__global__ void merge_topk_kernel(
    const c10::BFloat16* __restrict__ candidate_values,
    const int32_t* __restrict__ candidate_indices,
    c10::BFloat16* __restrict__ output_values,
    int64_t* __restrict__ output_indices,
    int64_t rows,
    int64_t candidate_width,
    int k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int* shared_positions = reinterpret_cast<int*>(shared_values + blockDim.x);
  int* selected = shared_positions + blockDim.x;

  const int tid = threadIdx.x;
  const int64_t candidate_base = row * candidate_width;
  const int64_t output_base = row * k;

  for (int out = 0; out < k; ++out) {
    float best_value = -CUDART_INF_F;
    int best_position = -1;

    for (int64_t pos = tid; pos < candidate_width; pos += blockDim.x) {
      bool already_selected = false;
      for (int prev = 0; prev < out; ++prev) {
        if (selected[prev] == static_cast<int>(pos)) {
          already_selected = true;
          break;
        }
      }
      if (already_selected) {
        continue;
      }

      const float value = static_cast<float>(candidate_values[candidate_base + pos]);
      const int latent_index = static_cast<int>(candidate_indices[candidate_base + pos]);
      const int position = static_cast<int>(pos);
      const int current_best_index =
          best_position < 0 ? -1 : static_cast<int>(candidate_indices[candidate_base + best_position]);

      if (value > best_value ||
          (value == best_value && (current_best_index < 0 || latent_index < current_best_index))) {
        best_value = value;
        best_position = position;
      }
    }

    shared_values[tid] = best_value;
    shared_positions[tid] = best_position;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = shared_values[tid + stride];
        const int other_position = shared_positions[tid + stride];
        const int this_position = shared_positions[tid];
        const int other_index = other_position < 0
            ? -1
            : static_cast<int>(candidate_indices[candidate_base + other_position]);
        const int this_index = this_position < 0
            ? -1
            : static_cast<int>(candidate_indices[candidate_base + this_position]);

        if (other_value > shared_values[tid] ||
            (other_value == shared_values[tid] && (this_index < 0 || other_index < this_index))) {
          shared_values[tid] = other_value;
          shared_positions[tid] = other_position;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      selected[out] = shared_positions[0];
      output_values[output_base + out] = c10::BFloat16(shared_values[0]);
      output_indices[output_base + out] =
          static_cast<int64_t>(candidate_indices[candidate_base + shared_positions[0]]);
    }
    __syncthreads();
  }
}

void cublaslt_linear_relu_block(
    cublasLtHandle_t handle,
    cudaStream_t stream,
    const torch::Tensor& input_2d,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    torch::Tensor& workspace,
    int64_t start,
    int64_t width) {
  const int64_t M = input_2d.size(0);
  const int64_t K = input_2d.size(1);

  cublasLtMatmulDesc_t op_desc;
  CUBLASLT_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t op_T = CUBLAS_OP_T;
  cublasOperation_t op_N = CUBLAS_OP_N;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_T, sizeof(op_T)));
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N)));

  cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_RELU_BIAS;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));

  const void* bias_ptr =
      static_cast<const void*>(bias.data_ptr<c10::BFloat16>() + start);
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));

  cudaDataType_t bias_dtype = CUDA_R_16BF;
  CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype)));

  cublasLtMatrixLayout_t layout_A;
  cublasLtMatrixLayout_t layout_B;
  cublasLtMatrixLayout_t layout_C;
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_16BF, K, width, K));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_16BF, K, M, K));
  CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_16BF, width, M, width));

  float alpha = 1.f;
  float beta = 0.f;
  const void* weight_ptr =
      static_cast<const void*>(weight.data_ptr<c10::BFloat16>() + start * K);

  CUBLASLT_CHECK(cublasLtMatmul(
      handle, op_desc,
      &alpha,
      weight_ptr, layout_A,
      input_2d.data_ptr(), layout_B,
      &beta,
      output.data_ptr(), layout_C,
      output.data_ptr(), layout_C,
      nullptr,
      workspace.data_ptr(), WORKSPACE_SIZE,
      stream));

  cublasLtMatrixLayoutDestroy(layout_C);
  cublasLtMatrixLayoutDestroy(layout_B);
  cublasLtMatrixLayoutDestroy(layout_A);
  cublasLtMatmulDescDestroy(op_desc);
}

std::tuple<torch::Tensor, torch::Tensor> linear_relu_topk_exact(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t k,
    int64_t block_n) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "linear_relu_topk_exact requires CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
  TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
  TORCH_CHECK(bias.scalar_type() == at::kBFloat16, "bias must be BF16");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
  TORCH_CHECK(k > 0, "k must be positive");
  TORCH_CHECK(block_n > 0, "block_n must be positive");

  const auto input_c = input.contiguous();
  const auto weight_c = weight.contiguous();
  const auto bias_c = bias.contiguous();

  const int64_t D = input_c.size(-1);
  const int64_t N = weight_c.size(0);
  const int64_t M = input_c.numel() / D;

  TORCH_CHECK(weight_c.size(1) == D, "weight/input dimension mismatch");
  TORCH_CHECK(bias_c.size(0) == N, "bias/weight dimension mismatch");
  TORCH_CHECK(D > 0, "input feature dimension must be positive");
  TORCH_CHECK(k <= N, "k must be <= d_sae");
  TORCH_CHECK(M <= INT32_MAX,
              "native fused exact top-k currently supports at most int32 row grid");
  TORCH_CHECK(N <= INT32_MAX,
              "native fused exact top-k currently supports at most int32 latent indices");
  TORCH_CHECK(k <= INT32_MAX,
              "native fused exact top-k currently supports at most int32 k");

  auto out_shape = input.sizes().vec();
  out_shape.back() = k;

  auto values_2d = torch::empty({M, k}, input_c.options());
  auto indices_2d = torch::empty({M, k}, input_c.options().dtype(at::kLong));
  if (M == 0) {
    return {values_2d.view(out_shape), indices_2d.view(out_shape)};
  }

  const int64_t num_blocks = (N + block_n - 1) / block_n;
  std::vector<int64_t> starts;
  std::vector<int64_t> widths;
  std::vector<int64_t> local_ks;
  std::vector<int64_t> offsets;
  starts.reserve(num_blocks);
  widths.reserve(num_blocks);
  local_ks.reserve(num_blocks);
  offsets.reserve(num_blocks);

  int64_t candidate_width = 0;
  for (int64_t start = 0; start < N; start += block_n) {
    const int64_t width = std::min(block_n, N - start);
    const int64_t local_k = std::min(k, width);
    TORCH_CHECK(width <= INT32_MAX,
                "native fused exact top-k currently supports at most int32 block width");
    starts.push_back(start);
    widths.push_back(width);
    local_ks.push_back(local_k);
    offsets.push_back(candidate_width);
    candidate_width += local_k;
  }
  TORCH_CHECK(candidate_width <= INT32_MAX,
              "native fused exact top-k currently supports at most int32 candidate width");

  auto input_2d = input_c.view({M, D});
  auto candidate_values = torch::empty({M, candidate_width}, input_c.options());
  auto candidate_indices = torch::empty({M, candidate_width}, input_c.options().dtype(at::kInt));
  auto workspace = at::empty(
      {static_cast<int64_t>(WORKSPACE_SIZE)},
      input_c.options().dtype(at::kByte));

  c10::cuda::CUDAGuard guard(input_c.device());
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  for (size_t block = 0; block < starts.size(); ++block) {
    auto block_acts = torch::empty({M, widths[block]}, input_c.options());
    cublaslt_linear_relu_block(
        handle,
        stream,
        input_2d,
        weight_c,
        bias_c,
        block_acts,
        workspace,
        starts[block],
        widths[block]);

    const int shared_bytes =
        TOPK_THREADS * static_cast<int>(sizeof(float) + sizeof(int)) +
        static_cast<int>(local_ks[block] * sizeof(int));
    local_topk_kernel<<<static_cast<unsigned int>(M), TOPK_THREADS, shared_bytes, stream>>>(
        block_acts.data_ptr<c10::BFloat16>(),
        candidate_values.data_ptr<c10::BFloat16>(),
        candidate_indices.data_ptr<int32_t>(),
        M,
        widths[block],
        candidate_width,
        offsets[block],
        starts[block],
        static_cast<int>(local_ks[block]));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int merge_shared_bytes =
      TOPK_THREADS * static_cast<int>(sizeof(float) + sizeof(int)) +
      static_cast<int>(k * sizeof(int));
  merge_topk_kernel<<<static_cast<unsigned int>(M), TOPK_THREADS, merge_shared_bytes, stream>>>(
      candidate_values.data_ptr<c10::BFloat16>(),
      candidate_indices.data_ptr<int32_t>(),
      values_2d.data_ptr<c10::BFloat16>(),
      indices_2d.data_ptr<int64_t>(),
      M,
      candidate_width,
      static_cast<int>(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {values_2d.view(out_shape), indices_2d.view(out_shape)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_relu_topk_exact",
      &linear_relu_topk_exact,
      "Exact fused Linear+ReLU+TopK using cuBLASLt block GEMMs and CUDA top-k");
}
