#include <sycl/sycl.hpp>
#include <torch/csrc/stable/tensor.h>

using namespace sycl;

void relu_xpu_impl(torch::stable::Tensor& output, const torch::stable::Tensor& input) {
    // Create SYCL queue directly
    sycl::queue queue;

    auto input_ptr = input.const_data_ptr<float>();
    auto output_ptr = output.mutable_data_ptr<float>();
    auto numel = input.numel();

    // Launch SYCL kernel
    queue.parallel_for(range<1>(numel), [=](id<1> idx) {
        auto i = idx[0];
        output_ptr[i] = input_ptr[i] > 0.0f ? input_ptr[i] : 0.0f;
    }).wait();
}

void relu(torch::stable::Tensor& out, const torch::stable::Tensor& input) {
    STD_TORCH_CHECK(input.device().type() == torch::headeronly::DeviceType::XPU, "input must be a XPU tensor");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Float,
                "Unsupported data type: ", input.scalar_type());

    STD_TORCH_CHECK(input.sizes().equals(out.sizes()),
                "Tensors must have the same shape.");

    STD_TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                "Tensors must have the same data type.");

    STD_TORCH_CHECK(input.device() == out.device(),
                "Tensors must be on the same device.");

    relu_xpu_impl(out, input);
}
