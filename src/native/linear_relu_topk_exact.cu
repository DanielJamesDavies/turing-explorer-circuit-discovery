#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> linear_relu_topk_exact(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t k,
    int64_t block_n) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
              "linear_relu_topk_exact native skeleton requires CUDA tensors");
  TORCH_CHECK(k > 0, "k must be positive");
  TORCH_CHECK(block_n > 0, "block_n must be positive");
  TORCH_CHECK(false,
              "linear_relu_topk_exact native CUDA kernel is a skeleton. "
              "Use the Python exact prototype until the CUDA kernel is implemented.");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_relu_topk_exact",
      &linear_relu_topk_exact,
      "Exact fused Linear+ReLU+TopK CUDA skeleton");
}
