"""Allclose variants that work around device limitations."""

import torch
from torch._prims_common import TensorLikeType


def fp8_allclose(
    a: TensorLikeType,
    b: TensorLikeType,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    """``torch.allclose`` replacement that handles FP8 types and MPS.

    On MPS (which lacks float64) the comparison is done in float32.
    Everywhere else the tensors are promoted to float64.
    """
    torch._refs._check_close_args(name="torch.allclose", a=a, b=b, rtol=rtol, atol=atol)

    if a.device.type == "mps" or b.device.type == "mps":
        a_cmp = a.float()
        b_cmp = b.float()
    else:
        a_cmp = a.double()
        b_cmp = b.double()

    return bool(
        torch.all(
            torch.isclose(a_cmp, b_cmp, rtol=rtol, atol=atol, equal_nan=equal_nan)
        ).item()
    )
