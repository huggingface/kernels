#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

// The shim's definition is guarded by USE_CUDA, so define here.
extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

#include <cmath>

__global__ void relu_kernel(float *__restrict__ out,
                            float const *__restrict__ input, const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    auto x = input[token_idx * d + idx];
    out[token_idx * d + idx] = x > 0.0f ? x : 0.0f;
  }
}

void relu(torch::stable::Tensor &out, torch::stable::Tensor const &input) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Float &&
                      out.scalar_type() == torch::headeronly::ScalarType::Float,
                  "relu_kernel only supports float32");

  STD_TORCH_CHECK(input.sizes().equals(out.sizes()),
                  "Tensors must have the same shape.");

  STD_TORCH_CHECK(input.scalar_type() == out.scalar_type(),
                  "Tensors must have the same data type.");

  STD_TORCH_CHECK(input.device() == out.device(),
                  "Tensors must be on the same device.");

  if (input.numel() == 0) {
    return;
  }

  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(input.get_device_index(), &stream_ptr));
  const cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
  relu_kernel<<<grid, block, 0, stream>>>(static_cast<float*>(out.data_ptr()),
                                          static_cast<const float*>(input.data_ptr()), d);
}
