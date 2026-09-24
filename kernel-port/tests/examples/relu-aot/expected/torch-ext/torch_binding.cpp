#include <torch/torch.h>

#include "registration.h"

void relu(torch::Tensor &out, const torch::Tensor &input);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("relu(Tensor! out, Tensor input) -> ()");
  ops.impl("relu", torch::kCPU, &relu);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
