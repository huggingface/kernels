#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu_fwd(Tensor input) -> Tensor");
  ops.def("relu_bwd(Tensor grad_output, Tensor input) -> Tensor");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("relu_fwd", torch::kCUDA, &relu_fwd);
  ops.impl("relu_bwd", torch::kCUDA, &relu_bwd);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)