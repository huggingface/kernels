import torch


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


__all__ = ["identity"]
