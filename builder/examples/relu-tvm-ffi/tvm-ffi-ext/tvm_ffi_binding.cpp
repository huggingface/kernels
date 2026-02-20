#include <tvm/ffi/tvm_ffi.h>

void relu_cuda(tvm::ffi::TensorView out, tvm::ffi::TensorView const input);

// TODO: CUDA ifdef
TVM_FFI_DLL_EXPORT_TYPED_FUNC(TVM_FFI_EXTENSION_NAME, relu_cuda);
