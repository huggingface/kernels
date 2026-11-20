import torch

from ._ops import ops


def conflicts(x: torch.Tensor) -> int:
    return ops.conflicts(x)


def conflicts_dynamic(x: torch.Tensor) -> int:
    return ops.conflicts_dynamic(x)


def conflicts_class(x: torch.Tensor) -> int:
    return ops.conflicts_class(x)
