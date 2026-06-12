import pytest
import relu_triton
import torch
import torch.nn.functional as F


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu(device, dtype):
    x = torch.randn(1024, 1024, dtype=dtype, device=device)
    torch.testing.assert_close(F.relu(x), relu_triton.relu(x))


@pytest.mark.kernels_ci
def test_relu_views(device):
    x = torch.arange(-20, 20, device=device, dtype=torch.float32)

    out = torch.empty_like(x)
    out_check = torch.empty_like(x)

    for i in range(41):
        out.fill_(42)
        out_check.fill_(42)

        relu_triton.relu(x[i:], out[i:])
        out_check[i:] = F.relu(x[i:])

        torch.testing.assert_close(out, out_check)


@pytest.mark.kernels_ci
def test_relu_layer(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu_triton.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))
