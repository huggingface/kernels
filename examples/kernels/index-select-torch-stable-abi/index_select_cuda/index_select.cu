#include <cstdint>
#include <torch_stable_abi_shim/csrc/stable/ops.h>

torch::stable::Tensor index_select_impl(
    const torch::stable::Tensor& input,
    int64_t dim,
    const torch::stable::Tensor& index) {
  return torch_stable_abi_shim::index_select(input, dim, index);
}
