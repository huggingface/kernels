import platform

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if platform.system() == "Darwin":
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
