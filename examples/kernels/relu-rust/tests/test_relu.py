import kernels
import pytest
import torch
import torch.nn.functional as F

relu_rust = kernels.get_kernel("kernels-test/relu-rust", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(F.relu(x), relu_rust.relu(x, torch.empty_like(x)))
