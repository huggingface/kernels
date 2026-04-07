import platform

import pytest
import torch


from relu_tvm_ffi._ops import has_jax


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

def pytest_runtest_setup(item):
    if "jax_only" in item.keywords and not has_jax:
        pytest.skip("skipping JAX-only test on host without JAX")
