#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("__KERNEL_NAME_NORMALIZED__(Tensor! out, Tensor input) -> ()");
#if defined(CPU_KERNEL)
  ops.impl("__KERNEL_NAME_NORMALIZED__", torch::kCPU, &__KERNEL_NAME_NORMALIZED__);
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("__KERNEL_NAME_NORMALIZED__", torch::kCUDA, &__KERNEL_NAME_NORMALIZED__);
#elif defined(METAL_KERNEL)
  ops.impl("__KERNEL_NAME_NORMALIZED__", torch::kMPS, __KERNEL_NAME_NORMALIZED__);
#elif defined(XPU_KERNEL)
  ops.impl("__KERNEL_NAME_NORMALIZED__", torch::kXPU, &__KERNEL_NAME_NORMALIZED__);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
