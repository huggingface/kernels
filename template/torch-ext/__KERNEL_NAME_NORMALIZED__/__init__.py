from typing import Optional

import torch

from ._ops import ops


def __KERNEL_NAME_NORMALIZED__(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.__KERNEL_NAME_NORMALIZED__(out, x)
    return out
