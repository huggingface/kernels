import torch

from ._ops import ops
from .op import _silu_and_mul


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return ops.silu_and_mul(x)


__all__ = ["silu_and_mul"]
