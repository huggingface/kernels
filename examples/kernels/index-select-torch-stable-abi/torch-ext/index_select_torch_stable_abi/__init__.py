import torch

from ._ops import ops


def index_select(input: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    return ops.index_select(input, dim, index)


__all__ = ["index_select"]
