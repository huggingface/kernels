import torch

from ._ops import ops


def invalid_cpp_symbol(input: torch.Tensor) -> torch.Tensor:
    return ops.invalid_cpp_symbol(input)


__all__ = ["invalid_cpp_symbol"]
