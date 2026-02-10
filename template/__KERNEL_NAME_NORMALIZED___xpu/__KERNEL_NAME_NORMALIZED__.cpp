#include <sycl/sycl.hpp>
#include <torch/torch.h>

void __KERNEL_NAME_NORMALIZED__(torch::Tensor& out, const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_xpu(), "input must be a XPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "only float32 supported");
    TORCH_CHECK(input.sizes() == out.sizes(), "Tensors must have same shape");
    TORCH_CHECK(input.scalar_type() == out.scalar_type(), "Tensors must have same dtype");
    TORCH_CHECK(input.device() == out.device(), "Tensors must be on same device");

    sycl::queue queue;
    auto input_ptr = input.data_ptr<float>();
    auto output_ptr = out.data_ptr<float>();
    auto n = input.numel();

    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        output_ptr[idx[0]] = input_ptr[idx[0]] + 1.0f;
    }).wait();
}
