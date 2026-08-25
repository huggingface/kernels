import torch
import kernels

kernel_deps = kernels.get_kernel("kernels-test/kernel-deps", version=1)


def test_swap_last_dimensions():
    torch.manual_seed(42)
    x = torch.randn(3, 4, 5, 6)
    y = kernel_deps.swap_last_dimensions(x)
    assert y.shape == (3, 4, 6, 5)
    torch.testing.assert_close(y, x.transpose(-1, -2))
