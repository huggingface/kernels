from typing import Optional

import torch

from . import layers
from ._ops import ops
from .op import _relu


def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.relu(out, x)
    return out


__all__ = ["layers", "relu"]
