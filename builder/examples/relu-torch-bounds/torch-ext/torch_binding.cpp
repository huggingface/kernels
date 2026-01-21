#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu(Tensor! out, Tensor input) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("relu", torch::kCUDA, &relu);
#elif defined(METAL_KERNEL)
  ops.impl("relu", torch::kMPS, relu);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
