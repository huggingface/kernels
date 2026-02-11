#include <torch/all.h>

void __KERNEL_NAME_NORMALIZED__(torch::Tensor &out, torch::Tensor const &input) {
    TORCH_CHECK(out.dtype() == torch::kFloat32, "Output tensor must be float32");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(out.numel() == input.numel(), "Tensors must have same size");

    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int64_t n = input.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] + 1.0f;
    }
}
