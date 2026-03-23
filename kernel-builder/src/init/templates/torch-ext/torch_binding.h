#pragma once

#include <torch/torch.h>

void {{ kernel_name_normalized }}(torch::Tensor &out, torch::Tensor const &input);
