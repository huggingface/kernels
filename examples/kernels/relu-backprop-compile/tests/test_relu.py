import platform

import pytest
import torch
import torch.nn.functional as F
from torch.library import opcheck

import relu_backprop_compile


def get_device():
    if platform.system() == "Darwin":
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cuda")


DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
]


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_forward(dtype):
    device = get_device()
    x = torch.randn(1024, 1024, dtype=dtype, device=device)
    expected = F.relu(x)
    actual = relu_backprop_compile.relu(x)
    torch.testing.assert_close(expected, actual)


def test_relu_gradient_numerical():
    device = get_device()
    x = torch.randn(32, 32, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(relu_backprop_compile.relu, x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_gradient_large_tensor(dtype):
    device = get_device()
    x = torch.randn(1024, 1024, dtype=dtype, device=device, requires_grad=True)
    y = relu_backprop_compile.relu(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    expected_grad = torch.where(
        x > 0,
        torch.tensor(1.0, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    )
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_gradient_comparison(dtype):
    device = get_device()
    x_data = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0], [0.5, -0.5, 1.5, -1.5, 0.0]],
        dtype=dtype,
        device=device,
    )

    x_kernel = x_data.clone().requires_grad_(True)
    y_kernel = relu_backprop_compile.relu(x_kernel)
    loss_custom = y_kernel.sum()
    loss_custom.backward()

    x_torch = x_data.clone().requires_grad_(True)
    y_torch = F.relu(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()

    torch.testing.assert_close(y_kernel, y_torch)
    torch.testing.assert_close(x_kernel.grad, x_torch.grad)


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_backward_chain(dtype):
    device = get_device()
    x = torch.randn(64, 128, dtype=dtype, device=device, requires_grad=True)
    y = relu_backprop_compile.relu(x)
    z = y * 2.0
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape

    expected_grad = torch.where(
        x > 0,
        torch.tensor(2.0, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    )
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (32, 64),
        (1, 1024),
        (128, 256),
        (8, 8, 16),
    ],
)
def test_relu_fwd_opcheck(shape, dtype):
    device = get_device()
    x = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    opcheck(relu_backprop_compile.ops.relu_fwd, (x,))


@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_torch_compile(dtype):
    device = get_device()

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1024, 1024)

        def forward(self, x):
            return relu_backprop_compile.relu(self.linear(x))

    model = SimpleModel().to(device).to(dtype)
    compiled_model = torch.compile(model, fullgraph=True)

    x = torch.randn((1024, 1024), dtype=dtype, device=device, requires_grad=True)

    y_original = model(x)
    y_compiled = compiled_model(x)

    torch.testing.assert_close(y_original, y_compiled)

    loss_original = y_original.sum()
    loss_compiled = y_compiled.sum()

    if x.grad is not None:
        x.grad.zero_()

    loss_original.backward(retain_graph=True)
    assert x.grad is not None
    grad_original = x.grad.clone()

    x.grad.zero_()
    loss_compiled.backward()
    assert x.grad is not None
    grad_compiled = x.grad.clone()

    torch.testing.assert_close(grad_original, grad_compiled)


@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_compile_recompilation_and_graph_break(dtype):
    device = get_device()

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 16)

        def forward(self, x):
            return relu_backprop_compile.relu(self.linear(x))

    model = SimpleModel().to(device).to(dtype)
    compiled_model = torch.compile(model, fullgraph=True)

    x = torch.randn((16, 16), dtype=dtype, device=device, requires_grad=True)
    with (
        torch._inductor.utils.fresh_inductor_cache(),
        torch._dynamo.config.patch(error_on_recompile=True),
    ):
        model(x)
        model(x)
