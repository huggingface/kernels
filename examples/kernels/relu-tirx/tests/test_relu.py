import pytest
import torch
import torch.nn.functional as F

import relu_tirx


@pytest.mark.kernels_ci
def test_relu(device):
    x = torch.randn(1024 * 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu_tirx.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_non_multiple_of_block(device):
    # The kernel guards the tail; sizes that are not a multiple of the
    # block size must work.
    x = torch.randn(1000, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu_tirx.relu(x, torch.empty_like(x)))
