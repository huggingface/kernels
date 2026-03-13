import pytest
import torch
import torch.nn.functional as F
from torch.library import opcheck

from silu_and_mul import ops, silu_and_mul


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_opcheck(device, requires_grad, dtype):
    torch.manual_seed(42)
    x = torch.randn(32, 128, device=device, requires_grad=requires_grad, dtype=dtype)
    opcheck(ops.silu_and_mul, (x,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [False, True])
# Only do float32, the numerical instabilities of float16 and bfloat16
# are too large with the different orderings of computing the gradients.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_silu_and_mul(device, requires_grad, dtype):
    torch.manual_seed(42)
    x_ref = torch.randn(
        32, 128, device=device, requires_grad=requires_grad, dtype=dtype
    )
    x = torch.empty(32, 128, device=device, requires_grad=requires_grad, dtype=dtype)
    with torch.no_grad():
        x.copy_(x_ref)

    y_ref = silu_and_mul_ref(x_ref)
    y = silu_and_mul(x)

    torch.testing.assert_close(y_ref, y)

    if requires_grad:
        d_y = torch.randn((32, 64), device=device, dtype=dtype)
        y_ref.backward(d_y)
        y.backward(d_y)
        torch.testing.assert_close(x_ref.grad, x.grad)
