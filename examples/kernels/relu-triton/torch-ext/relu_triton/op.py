import torch
import triton
import triton.language as tl

from ._ops import add_op_namespace_prefix


@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


@torch.library.custom_op(add_op_namespace_prefix("relu"), mutates_args={"out"})
def _relu(out: torch.Tensor, x: torch.Tensor) -> None:
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)


@_relu.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass
