import kernels
import pytest
import torch
import torch.nn.functional as F

relu_rust_cuda = kernels.get_kernel("kernels-test/relu-rust-cuda", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(
        F.relu(x), relu_rust_cuda.relu(x, torch.empty_like(x))
    )


@pytest.mark.kernels_ci
def test_rejects_mismatched_output():
    x = torch.randn(16, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception, match="same number of elements"):
        relu_rust_cuda.relu(x, torch.empty(32, dtype=torch.float32, device="cuda"))
