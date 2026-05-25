#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> linear_relu_topk_blockwise(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t k,
    int64_t block_n) {
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 2, "weight must be [d_sae, d_model]");
  TORCH_CHECK(bias.dim() == 1, "bias must be [d_sae]");
  TORCH_CHECK(k > 0, "k must be positive");
  TORCH_CHECK(block_n > 0, "block_n must be positive");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "input/weight dtype mismatch");
  TORCH_CHECK(weight.scalar_type() == bias.scalar_type(), "weight/bias dtype mismatch");
  TORCH_CHECK(input.device() == weight.device() && weight.device() == bias.device(),
              "input, weight, and bias must be on the same device");

  const auto d_model = input.size(-1);
  const auto d_sae = weight.size(0);
  TORCH_CHECK(weight.size(1) == d_model, "weight.shape[1] must match input.shape[-1]");
  TORCH_CHECK(bias.size(0) == d_sae, "bias.shape[0] must match weight.shape[0]");
  TORCH_CHECK(k <= d_sae, "k must be <= d_sae");

  std::vector<int64_t> input_shape = input.sizes().vec();
  const int64_t rows = input.numel() / d_model;
  auto input_2d = input.reshape({rows, d_model});

  std::vector<torch::Tensor> value_chunks;
  std::vector<torch::Tensor> index_chunks;
  for (int64_t start = 0; start < d_sae; start += block_n) {
    const int64_t end = std::min(start + block_n, d_sae);
    const int64_t local_k = std::min(k, end - start);
    auto weight_chunk = weight.slice(/*dim=*/0, start, end);
    auto bias_chunk = bias.slice(/*dim=*/0, start, end);
    auto acts = torch::relu(torch::matmul(input_2d, weight_chunk.t()) + bias_chunk);
    auto topk = torch::topk(acts, local_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
    value_chunks.push_back(std::get<0>(topk));
    index_chunks.push_back(std::get<1>(topk) + start);
  }

  auto candidates = torch::cat(value_chunks, /*dim=*/-1);
  auto candidate_indices = torch::cat(index_chunks, /*dim=*/-1);
  auto final_topk = torch::topk(candidates, k, /*dim=*/-1, /*largest=*/true, /*sorted=*/false);
  auto values = std::get<0>(final_topk);
  auto positions = std::get<1>(final_topk);
  auto indices = candidate_indices.gather(/*dim=*/-1, positions).to(torch::kInt64);

  input_shape.back() = k;
  return {
      values.reshape(input_shape),
      indices.reshape(input_shape),
  };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_relu_topk_blockwise",
      &linear_relu_topk_blockwise,
      "Blockwise Linear+ReLU+TopK prototype");
}
