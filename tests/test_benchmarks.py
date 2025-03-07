import pytest
import torch

from kernels import get_kernel


@pytest.fixture
def kernel():
    return get_kernel("kernels-community/activation")


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    return "cuda"


def test_gelu_small(kernel, device, benchmark):
    x = torch.randn(32, 32, dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    benchmark(kernel.gelu_fast, y, x)


def test_gelu_medium(kernel, device, benchmark):
    x = torch.randn(128, 128, dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    benchmark(kernel.gelu_fast, y, x)


def test_gelu_large(kernel, device, benchmark):
    x = torch.randn(512, 512, dtype=torch.float16, device=device)
    y = torch.empty_like(x)
    benchmark(kernel.gelu_fast, y, x)
