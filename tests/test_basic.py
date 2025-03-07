import pytest
import torch

from kernels import get_kernel


@pytest.fixture
def kernel():
    return get_kernel("kernels-community/activation")


@pytest.fixture
def universal_kernel():
    return get_kernel("kernels-community/triton-scaled-mm")


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


def test_universal_kernel(universal_kernel):
    torch.manual_seed(0)
    A = torch.randint(-10, 10, (64, 128), dtype=torch.int8, device="cuda")
    B = torch.randint(-10, 10, (128, 96), dtype=torch.int8, device="cuda")
    scale_a = torch.tensor(0.4, dtype=torch.float16, device="cuda")
    scale_b = torch.tensor(0.6, dtype=torch.float16, device="cuda")

    out = universal_kernel.triton_scaled_mm(A, B, scale_a, scale_b, torch.float16)
    out_check = (A * scale_a) @ (B * scale_b)
    out_check = out_check.to(torch.float16)

    torch.testing.assert_close(out, out_check, rtol=1e-1, atol=1e-1)
