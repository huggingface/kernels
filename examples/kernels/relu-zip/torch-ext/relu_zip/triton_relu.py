import functools
import importlib.resources
import json
import math
from typing import Optional

import torch
import triton
import triton.language as tl

# Launch configuration that is used when there are no tuned configurations for
# the device.
DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}

# Tuned configurations are stored per device, see `scripts/tune_triton.py`.
_CONFIG_DIR = "configs"


@triton.jit
def _relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def _device_name(device: torch.device) -> str:
    """Device name that is safe to use in file names, e.g. `NVIDIA_L4`."""
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
    return "".join(c if c.isalnum() else "_" for c in name)


@functools.cache
def _load_tuned_configs(device_name: str) -> dict:
    """Load tuned configurations: `{dtype: {n_elements: config}}`."""
    path = importlib.resources.files(__package__) / _CONFIG_DIR / f"{device_name}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())["configs"]


def select_config(x: torch.Tensor) -> dict:
    """Select the launch configuration for `x`."""
    tuned = _load_tuned_configs(_device_name(x.device)).get(
        str(x.dtype).removeprefix("torch.")
    )
    if not tuned:
        return DEFAULT_CONFIG

    # Sizes are powers of two, so compare them on a log scale.
    closest = min(
        tuned, key=lambda n: abs(math.log2(int(n)) - math.log2(max(x.numel(), 1)))
    )
    return tuned[closest]


def relu_triton(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    config: Optional[dict] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(x)
    if config is None:
        config = select_config(x)

    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _relu_kernel[grid](x, out, n_elements, **config)
    return out
