#include <sycl/sycl.hpp>
#include <torch/torch.h>

using namespace sycl;

#ifndef WHO_AM_I_IF_NOT_THE_CANARY
#error "Kernel flags are not correctly handled."
#endif

void relu_xpu_impl(torch::Tensor& output, const torch::Tensor& input) {
    // Create SYCL queue directly
    sycl::queue queue;

    auto input_ptr = input.data_ptr<float>();
    auto output_ptr = output.data_ptr<float>();
    auto numel = input.numel();

    // Launch SYCL kernel
    queue.parallel_for(range<1>(numel), [=](id<1> idx) {
        auto i = idx[0];
        output_ptr[i] = input_ptr[i] > 0.0f ? input_ptr[i] : 0.0f;
    }).wait();
}

void relu(torch::Tensor& out, const torch::Tensor& input) {
    TORCH_CHECK(input.device().is_xpu(), "input must be a XPU tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
                "Unsupported data type: ", input.scalar_type());

    TORCH_CHECK(input.sizes() == out.sizes(),
                "Tensors must have the same shape. Got input shape: ",
                input.sizes(), " and output shape: ", out.sizes());

    TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                "Tensors must have the same data type. Got input dtype: ",
                input.scalar_type(), " and output dtype: ", out.scalar_type());

    TORCH_CHECK(input.device() == out.device(),
                "Tensors must be on the same device. Got input device: ",
                input.device(), " and output device: ", out.device());

    relu_xpu_impl(out, input);
}
