#pragma once

#include <torch/csrc/stable/tensor.h>

void relu(torch::stable::Tensor &out, torch::stable::Tensor const &input);
