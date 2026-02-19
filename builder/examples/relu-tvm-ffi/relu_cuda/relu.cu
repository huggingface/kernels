#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/extra/c_env_api.h>

using namespace tvm;

#define CHECK_CUDA(x) \
  TVM_FFI_CHECK((x).device().device_type == kDLCUDA, ValueError) << #x " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) \
  TVM_FFI_CHECK((x).IsContiguous(), ValueError) << #x " must be contiguous"
#define CHECK_INPUT(x)   \
  do {                   \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x); \
  } while (0)
#define CHECK_DEVICE(a, b)                                                          \
  do {                                                                              \
    TVM_FFI_CHECK((a).device().device_type == (b).device().device_type, ValueError) \
        << #a " and " #b " must be on the same device type";                        \
    TVM_FFI_CHECK((a).device().device_id == (b).device().device_id, ValueError)     \
        << #a " and " #b " must be on the same device";                             \
  } while (0)

constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};

__global__ void relu_kernel(float *__restrict__ out,
                            float const *__restrict__ input, const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    auto x = input[token_idx * d + idx];
    out[token_idx * d + idx] = x > 0.0f ? x : 0.0f;
  }
}

void relu_cuda(ffi::TensorView out, ffi::TensorView const input) {
  CHECK_INPUT(input);
  CHECK_INPUT(out);
  CHECK_DEVICE(input, out);

  TVM_FFI_CHECK(input.dtype() == out.dtype(), ValueError) << "input/output dtype mismatch";
  TVM_FFI_CHECK(input.numel() == out.numel(), ValueError) << "input/output size mismatch";

  ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));

  int d = input.size(-1);
  int64_t num_tokens = input.numel() / d;
  int64_t block(std::min(d, 1024));

  if (input.dtype() == dl_float32) {
    relu_kernel<<<num_tokens, block, 0, stream>>>(static_cast<float*>(out.data_ptr()),
                                                  static_cast<const float*>(input.data_ptr()),
                                                  d);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported dtype: " << input.dtype();
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(relu_cuda, relu_cuda);
