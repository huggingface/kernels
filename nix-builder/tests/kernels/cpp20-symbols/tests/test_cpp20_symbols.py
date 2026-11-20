import kernels
import pytest
import torch

cpp20_symbols = kernels.get_kernel("kernels-test/cpp20-symbols", version=1)


@pytest.mark.kernels_ci
def test_float_to_chars_runs():
    x = torch.tensor([3.14], dtype=torch.float64)
    out = cpp20_symbols.float_to_chars(x)
    torch.testing.assert_close(out, x)


@pytest.mark.kernels_ci
def test_float_to_chars_float32():
    x = torch.tensor([2.71828], dtype=torch.float32)
    out = cpp20_symbols.float_to_chars(x)
    torch.testing.assert_close(out, x)
