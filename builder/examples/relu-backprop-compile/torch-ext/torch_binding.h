#pragma once

#include <torch/torch.h>

torch::Tensor relu_fwd(torch::Tensor const &input);
torch::Tensor relu_bwd(torch::Tensor const &grad_output, torch::Tensor const &input);