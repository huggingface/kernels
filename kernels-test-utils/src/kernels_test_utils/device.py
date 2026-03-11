"""Device detection utilities for kernel tests."""

from typing import List

import pytest
import torch


def get_device() -> torch.device:
    """Return the best available compute device (MPS > CUDA > XPU > CPU)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def get_available_devices() -> List[str]:
    """Return device strings suitable for pytest parametrization.

    On MPS: ``["mps"]``
    On CUDA: ``["cuda:0", "cuda:1", ...]`` for each visible GPU.
    On XPU: ``["xpu:0", "xpu:1", ...]`` for each visible accelerator.
    Fallback: ``["cpu"]``
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ["mps"]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(max(1, torch.cuda.device_count()))]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return [f"xpu:{i}" for i in range(max(1, torch.xpu.device_count()))]
    return ["cpu"]


def skip_if_no_gpu() -> None:
    """Call inside a test to skip when no GPU is available."""
    dev = get_device()
    if dev.type == "cpu":
        pytest.skip("No GPU device available")
