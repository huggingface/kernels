#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cutlass_gemm(Tensor! out, Tensor A, Tensor B) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("cutlass_gemm", torch::kCUDA, &cutlass_gemm);
#elif defined(XPU_KERNEL)
  ops.impl("cutlass_gemm", torch::kXPU, &cutlass_gemm);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
