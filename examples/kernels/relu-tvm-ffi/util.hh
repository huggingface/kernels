#include <tvm/ffi/tvm_ffi.h>

#define CHECK_CUDA(x) \
  TVM_FFI_CHECK((x).device().device_type == kDLCUDA, ValueError) << #x " must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) \
  TVM_FFI_CHECK((x).IsContiguous(), ValueError) << #x " must be contiguous"
#define CHECK_INPUT(x)   \
  do {                   \
    CHECK_CONTIGUOUS(x); \
  } while (0)
#define CHECK_INPUT_CUDA(x)   \
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


