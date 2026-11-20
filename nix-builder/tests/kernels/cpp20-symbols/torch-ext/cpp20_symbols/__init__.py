import torch

from ._ops import ops


def float_to_chars(input: torch.Tensor) -> torch.Tensor:
    return ops.float_to_chars(input)


__all__ = ["float_to_chars"]
