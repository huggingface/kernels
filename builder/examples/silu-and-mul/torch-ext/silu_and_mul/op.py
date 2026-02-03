import torch
import torch.nn.functional as F

from ._ops import add_op_namespace_prefix


@torch.library.custom_op(add_op_namespace_prefix("silu_and_mul"), mutates_args=())
def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def backward(ctx, grad_output):
    x = ctx.saved_tensors[0]
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    sigmoid_x1 = torch.sigmoid(x1)
    silu_x1 = F.silu(x1)
    dsilu_dx1 = sigmoid_x1 + silu_x1 * (1 - sigmoid_x1)
    dx1 = grad_output * x2 * dsilu_dx1
    dx2 = grad_output * silu_x1
    return torch.cat([dx1, dx2], dim=-1)


def setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


_silu_and_mul.register_autograd(backward, setup_context=setup_context)


@_silu_and_mul.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[0], x.shape[1] // 2)
