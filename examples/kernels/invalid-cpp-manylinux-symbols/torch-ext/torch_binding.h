#pragma once

#include <torch/torch.h>

// Uses std::to_chars for floating-point, which requires GLIBCXX_3.4.29
// (introduced in GCC 11). This exceeds the GLIBCXX_3.4.25 ceiling of
// manylinux_2_28 target systems and should be caught by the ABI check.
torch::Tensor invalid_cpp_symbol(torch::Tensor const &input);
