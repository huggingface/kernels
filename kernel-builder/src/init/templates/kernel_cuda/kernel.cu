#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

__global__ void {{ kernel_name_normalized }}_kernel(float *__restrict__ out,
                            float const *__restrict__ input, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = input[idx] + 1.0f;
  }
}

void {{ kernel_name_normalized }}(torch::Tensor &out, torch::Tensor const &input) {
  TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
              "{{ kernel_name_normalized }} only supports float32");
  TORCH_CHECK(input.sizes() == out.sizes(),
              "Tensors must have the same shape");
  TORCH_CHECK(input.scalar_type() == out.scalar_type(),
              "Tensors must have the same dtype");
  TORCH_CHECK(input.device() == out.device(),
              "Tensors must be on the same device");

  int n = input.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  {{ kernel_name_normalized }}_kernel<<<blocks, threads, 0, stream>>>(
      out.data_ptr<float>(), input.data_ptr<float>(), n);
}
