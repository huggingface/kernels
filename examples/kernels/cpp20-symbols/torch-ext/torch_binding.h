#pragma once

#include <torch/torch.h>

// Uses std::to_chars for floating-point, which requires GLIBCXX_3.4.29
// (introduced in GCC 11). We use this to verify that manylinux_2_28
// kernels build correctly: the Red Hat toolset statically links the newer
// libstdc++ symbols that exceed the system GLIBCXX_3.4.25 ceiling of
// AlmaLinux 8 / RHEL 8.
torch::Tensor float_to_chars(torch::Tensor const &input);
