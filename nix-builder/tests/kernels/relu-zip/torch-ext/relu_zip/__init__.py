from typing import Optional

import torch

from ._ops import ops

from . import layers

# Triton is not available in all build environments (e.g. CPU and Metal).
try:
    import triton  # noqa: F401
except ImportError:
    relu_triton = None
else:
    from .triton_relu import relu_triton


def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.relu(out, x)
    return out


__all__ = ["relu", "layers"]
if relu_triton is not None:
    __all__.append("relu_triton")
