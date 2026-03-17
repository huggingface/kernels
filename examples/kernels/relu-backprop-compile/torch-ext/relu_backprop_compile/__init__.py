import torch
from torch.library import register_autograd

from ._ops import ops, add_op_namespace_prefix


@torch.library.register_fake(add_op_namespace_prefix("relu_fwd"))
def relu_fwd_fake(input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


@torch.library.register_fake(add_op_namespace_prefix("relu_bwd"))
def relu_bwd_fake(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(input)


def relu(x: torch.Tensor) -> torch.Tensor:
    return ops.relu_fwd(x)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad_outputs: torch.Tensor):
    (input,) = ctx.saved_tensors
    grad_outputs_contiguous = grad_outputs.contiguous()
    return ops.relu_bwd(grad_outputs_contiguous, input)


register_autograd(
    add_op_namespace_prefix("relu_fwd"), _backward, setup_context=_setup_context
)
