#include <torch/extension.h>

void relu(torch::Tensor &out, const torch::Tensor &input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("relu", &relu); }
