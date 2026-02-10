import platform

import torch

import __KERNEL_NAME_NORMALIZED__


def test___KERNEL_NAME_NORMALIZED__():
    if platform.system() == "Darwin":
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    expected = x + 1.0
    result = __KERNEL_NAME_NORMALIZED__.__KERNEL_NAME_NORMALIZED__(x)
    torch.testing.assert_close(result, expected)
