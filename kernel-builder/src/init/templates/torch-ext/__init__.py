from typing import Optional

import torch

from ._ops import ops


def {{ kernel_name_normalized }}(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.{{ kernel_name_normalized }}(out, x)
    return out
