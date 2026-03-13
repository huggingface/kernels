#pragma once

#include <torch/torch.h>

void cutlass_gemm(torch::Tensor &out, torch::Tensor const &A, torch::Tensor const &B);
