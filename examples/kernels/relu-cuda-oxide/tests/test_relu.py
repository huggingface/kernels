import kernels
import pytest
import torch
import torch.nn.functional as F

relu_cuda_oxide = kernels.get_kernel("kernels-test/relu-cuda-oxide", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(F.relu(x), relu_cuda_oxide.relu(x, torch.empty_like(x)))
