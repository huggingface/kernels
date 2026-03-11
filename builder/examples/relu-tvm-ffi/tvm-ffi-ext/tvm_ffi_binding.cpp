#include <tvm/ffi/tvm_ffi.h>

#ifdef CUDA_KERNEL
void relu_cuda(tvm::ffi::TensorView out, tvm::ffi::TensorView const input);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(relu_cuda, relu_cuda);
#endif

#ifdef CPU_KERNEL
void relu_cpu(tvm::ffi::TensorView out, tvm::ffi::TensorView const input);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(relu_cpu, relu_cpu);
#endif
