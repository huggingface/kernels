#include <torch/torch.h>

void relu(torch::Tensor &out, const torch::Tensor &input) {
  TORCH_CHECK(input.device().is_cpu() && out.device().is_cpu());
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 && out.scalar_type() == torch::kFloat32);
  TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(input.sizes() == out.sizes());
  const auto *x = input.data_ptr<float>();
  auto *y = out.data_ptr<float>();
  for (int64_t i = 0; i < input.numel(); ++i) {
    y[i] = x[i] < 0.0f ? 0.0f : x[i];
  }
}
