import pytest
import torch
import torch.nn.functional as F

import relu


@pytest.mark.kernels_ci
def test_relu(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_views(device):
    x = torch.arange(-20, 20, device=device, dtype=torch.float32)

    # Keep buffers and fill on each iteration. Stable pointers make C++-side
    # pointer inspection esier.
    out = torch.empty_like(x)
    out_check = torch.empty_like(x)

    for i in range(41):
        # Put a sentineal value in the output.
        out.fill_(42)
        out_check.fill_(42)

        relu.relu(x[i:], out[i:])
        out_check[i:] = F.relu(x[i:])

        torch.testing.assert_close(out, out_check)


@pytest.mark.kernels_ci
def test_relu_layer(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))
