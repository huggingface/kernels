from typing import Optional

import torch

from ._ops import ops
from .gemm import _gemm  # noqa: F401 — imported to register the op.
from .tuning import tune_gemm

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


def gemm(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute ``a @ b`` using a Triton GEMM kernel with tuned configurations."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible GEMM shapes: {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype not in _SUPPORTED_DTYPES or a.dtype != b.dtype:
        raise ValueError(f"gemm expects two {_SUPPORTED_DTYPES} tensors of the same dtype")
    if out is None:
        out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
    ops.gemm(out, a, b)
    return out


__all__ = ["gemm", "tune_gemm"]
