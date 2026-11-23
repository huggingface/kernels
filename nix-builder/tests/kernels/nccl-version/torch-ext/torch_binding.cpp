#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("nccl_header_version", &nccl_header_version);
  ops.def("nccl_runtime_version", &nccl_runtime_version);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
