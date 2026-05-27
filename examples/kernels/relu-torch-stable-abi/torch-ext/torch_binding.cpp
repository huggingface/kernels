#include <torch/csrc/stable/library.h>

#include "registration.h"
#include "torch_binding.h"

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu(Tensor! out, Tensor input) -> ()");
}

#if defined(CPU_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CPU, ops) {
  ops.impl("relu", TORCH_BOX(&relu));
}
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("relu", TORCH_BOX(&relu));
}
#elif defined(METAL_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, MPS, ops) {
  ops.impl("relu", TORCH_BOX(&relu));
}
#elif defined(XPU_KERNEL)
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, XPU, ops) {
  ops.impl("relu", TORCH_BOX(&relu));
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
