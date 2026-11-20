import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        pytest.skip("Neither CUDA nor XPU device available")
