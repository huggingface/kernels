#pragma once

#include <cstdint>
#include <torch/csrc/stable/tensor.h>

torch::stable::Tensor index_select_impl(
    const torch::stable::Tensor& input,
    int64_t dim,
    const torch::stable::Tensor& index);
