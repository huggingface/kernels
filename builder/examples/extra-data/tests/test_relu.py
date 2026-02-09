import platform

import torch
import torch.nn.functional as F

import extra_data


def test_relu():
    if platform.system() == "Darwin":
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_allclose(F.relu(x), extra_data.relu(x))


def test_relu_layer():
    if platform.system() == "Darwin":
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = extra_data.layers.ReLU()
    torch.testing.assert_allclose(F.relu(x), layer(x))


def test_data():
    assert extra_data.EASTER_EGG == 42
