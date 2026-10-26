#include <torch/csrc/stable/library.h>

#include "registration.h"
#include "torch_binding.h"

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("index_select(Tensor input, int dim, Tensor index) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("index_select", TORCH_BOX(&index_select_impl));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
