#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("conflicts(Tensor x) -> int");
  ops.def("conflicts_dynamic(Tensor x) -> int");
  ops.def("conflicts_class(Tensor x) -> int");
#if defined(CPU_KERNEL)
  ops.impl("conflicts", torch::kCPU, &test::conflicts);
  ops.impl("conflicts_dynamic", torch::kCPU, &test::conflicts_dynamic);
  ops.impl("conflicts_class", torch::kCPU, &test::conflicts_class);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
