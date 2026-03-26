#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("{{ kernel_name_normalized }}(Tensor! out, Tensor input) -> ()");
#if defined(CPU_KERNEL)
  ops.impl("{{ kernel_name_normalized }}", torch::kCPU, &{{ kernel_name_normalized }});
#elif defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("{{ kernel_name_normalized }}", torch::kCUDA, &{{ kernel_name_normalized }});
#elif defined(METAL_KERNEL)
  ops.impl("{{ kernel_name_normalized }}", torch::kMPS, {{ kernel_name_normalized }});
#elif defined(XPU_KERNEL)
  ops.impl("{{ kernel_name_normalized }}", torch::kXPU, &{{ kernel_name_normalized }});
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
