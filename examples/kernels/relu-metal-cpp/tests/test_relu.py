import platform

import kernels
import pytest
import torch
import torch.nn.functional as F

relu = kernels.get_kernel("kernels-test/relu-metal-cpp", version=1)


@pytest.mark.kernels_ci
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
