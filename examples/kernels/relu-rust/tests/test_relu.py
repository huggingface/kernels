import pytest
import torch
import torch.nn.functional as F

import relu_rust


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(F.relu(x), relu_rust.relu(x, torch.empty_like(x)))
