import json
from pathlib import Path
from typing import Optional

import torch

from ._ops import ops

from . import layers


# This is the regular ReLU, but this example also shows how to embed some
# non-Python data. This can be used for e.g. Triton tuning data.


def _read_json() -> dict:
    json_path = Path(__file__).parent / "data.json"
    with open(json_path, "r") as f:
        return json.load(f)


EASTER_EGG = _read_json()


def relu(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    ops.relu(out, x)
    return out


__all__ = ["EASTER_EGG", "relu", "layers"]
