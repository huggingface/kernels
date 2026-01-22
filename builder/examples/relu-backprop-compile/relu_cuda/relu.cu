#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cmath>

template<typename T>
__global__ void relu_fwd_kernel(T *__restrict__ out,
                                T const *__restrict__ input, int const d) {
  int64_t const token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    int64_t const offset = token_idx * d + idx;
    auto x = input[offset];
    // Casting to float because `>` cannot be resolved BFloat16.
    out[offset] = static_cast<float>(x) > 0.0f ? x : static_cast<T>(0);
  }
}

template<typename T>
__global__ void relu_bwd_kernel(T *__restrict__ grad_input,
                                T const *__restrict__ grad_output,
                                T const *__restrict__ input, int const d) {
  int64_t const token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    int64_t const offset = token_idx * d + idx;
    auto x = static_cast<float>(input[offset]);
    auto grad_out = grad_output[offset];
    grad_input[offset] = static_cast<float>(x) > 0.0f ? grad_out : static_cast<T>(0);
  }
}

torch::Tensor relu_fwd(torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
              input.scalar_type() == at::ScalarType::Double ||
              input.scalar_type() == at::ScalarType::Half ||
              input.scalar_type() == at::ScalarType::BFloat16,
              "relu_kernel supports float32, float64, float16, and bfloat16");

  torch::Tensor out = torch::empty_like(input);

  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  at::cuda::OptionalCUDAGuard const device_guard(device_of(input));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  
  // Dispatch based on scalar type
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "relu_fwd_kernel", [&] {
    relu_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(), 
        d);
  });
  
  return out;
}

torch::Tensor relu_bwd(torch::Tensor const &grad_output, torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  
  // Check for supported data types
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
              input.scalar_type() == at::ScalarType::Double ||
              input.scalar_type() == at::ScalarType::Half ||
              input.scalar_type() == at::ScalarType::BFloat16,
              "relu_bwd_kernel supports float32, float64, float16, and bfloat16");
  
  TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  TORCH_CHECK(grad_output.scalar_type() == input.scalar_type(),
              "grad_output and input must have the same data type");
  
  TORCH_CHECK(input.sizes() == grad_output.sizes(),
              "input and grad_output must have the same shape");
  
  TORCH_CHECK(input.device() == grad_output.device(),
              "input and grad_output must be on the same device");

  torch::Tensor grad_input = torch::empty_like(input);

  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  at::cuda::OptionalCUDAGuard const device_guard(device_of(input));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "relu_bwd_kernel", [&] {
    relu_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
        grad_input.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(), 
        d);
  });
  
  return grad_input;
}
