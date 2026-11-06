import kernels
import pytest
import torch
import torch.nn.functional as F

relu_rust_cutile = kernels.get_kernel("kernels-test/relu-rust-cutile", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(F.relu(x), relu_rust_cutile.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_partial_tile():
    x = torch.randn(1000, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(F.relu(x), relu_rust_cutile.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_rejects_mismatched_output():
    x = torch.randn(16, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception, match="same number of elements"):
        relu_rust_cutile.relu(x, torch.empty(32, dtype=torch.float32, device="cuda"))
