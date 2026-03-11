import torch

from kernels_test_utils import get_device

import __KERNEL_NAME_NORMALIZED__


def test___KERNEL_NAME_NORMALIZED__():
    device = get_device()

    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    expected = x + 1.0
    result = __KERNEL_NAME_NORMALIZED__.__KERNEL_NAME_NORMALIZED__(x)
    torch.testing.assert_close(result, expected)
