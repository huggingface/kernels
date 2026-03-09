"""Default tolerance tables for kernel tests."""

from typing import Dict

import torch

DEFAULT_TOLERANCES: Dict[torch.dtype, Dict[str, float]] = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1.6e-2},
}


def get_tolerances(dtype: torch.dtype) -> Dict[str, float]:
    """Return ``{"atol": ..., "rtol": ...}`` for *dtype*.

    Falls back to ``atol=0.1, rtol=0.1`` for unknown dtypes.
    """
    return DEFAULT_TOLERANCES.get(dtype, {"atol": 0.1, "rtol": 0.1})
