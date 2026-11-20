#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("float_to_chars(Tensor input) -> Tensor");
#if defined(CPU_KERNEL)
    ops.impl("float_to_chars", torch::kCPU, &float_to_chars);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
