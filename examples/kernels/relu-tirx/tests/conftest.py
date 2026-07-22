import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("TIRx kernels require CUDA")
    return torch.device("cuda")
