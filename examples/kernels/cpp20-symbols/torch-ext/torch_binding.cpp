#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("invalid_cpp_symbol(Tensor input) -> Tensor");
#if defined(CPU_KERNEL)
    ops.impl("invalid_cpp_symbol", torch::kCPU, &invalid_cpp_symbol);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
