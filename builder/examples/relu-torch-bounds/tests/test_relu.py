import platform

import torch
import torch.nn.functional as F

import relu


def test_relu():
    if platform.system() == "Darwin":
        device = torch.device("mps")
    else:
        device = torch.device("cuda")
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_allclose(F.relu(x), relu.relu(x))
