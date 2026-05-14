import cpp20_symbols
import pytest
import torch


@pytest.mark.kernels_ci
def test_invalid_cpp_symbol_runs():
    x = torch.tensor([3.14], dtype=torch.float64)
    out = cpp20_symbols.invalid_cpp_symbol(x)
    torch.testing.assert_close(out, x)


@pytest.mark.kernels_ci
def test_invalid_cpp_symbol_float32():
    x = torch.tensor([2.71828], dtype=torch.float32)
    out = cpp20_symbols.invalid_cpp_symbol(x)
    torch.testing.assert_close(out, x)
