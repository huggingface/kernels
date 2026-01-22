import platform

import torch
import torch.nn.functional as F

import relu


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
    torch.testing.assert_allclose(F.relu(x), relu.relu(x))
