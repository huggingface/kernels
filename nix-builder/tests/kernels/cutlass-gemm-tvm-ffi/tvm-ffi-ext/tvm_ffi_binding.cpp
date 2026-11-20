#include <tvm/ffi/tvm_ffi.h>

void cutlass_gemm(tvm::ffi::TensorView out, tvm::ffi::TensorView const A, tvm::ffi::TensorView const B);

#if defined(CUDA_KERNEL) || defined(XPU_KERNEL)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cutlass_gemm, cutlass_gemm);
#endif