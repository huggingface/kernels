#pragma once

#include <torch/torch.h>

namespace test {

int64_t conflicts(torch::Tensor const &input);
int64_t conflicts_dynamic(torch::Tensor const &input);
int64_t conflicts_class(torch::Tensor const &input);

}
