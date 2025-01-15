import pytest
import torch
from hf_kernels import get_kernel


@pytest.fixture
def kernel():
    return get_kernel("kernels-community/activation")


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    return "cuda"


def test_gelu_fast(kernel, device):
    x = torch.arange(1, 10, dtype=torch.float16, device=device).view(3, 3)
    y = torch.empty_like(x)

    kernel.gelu_fast(y, x)

    expected = torch.tensor(
        [[0.8408, 1.9551, 2.9961], [4.0000, 5.0000, 6.0000], [7.0000, 8.0000, 9.0000]],
        device=device,
        dtype=torch.float16,
    )

    assert torch.allclose(y, expected)
