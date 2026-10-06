import ctypes
import sys

import kernels
import pytest
import torch
import torch.nn.functional as F

relu_rust = kernels.get_kernel("kernels-test/relu-rust", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cpu")
    torch.testing.assert_close(F.relu(x), relu_rust.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_rejects_mismatched_output():
    x = torch.randn(16, dtype=torch.float32, device="cpu")
    with pytest.raises(Exception, match="same number of elements"):
        relu_rust.relu(x, torch.empty(32, dtype=torch.float32, device="cpu"))


@pytest.mark.kernels_ci
@pytest.mark.skipif(sys.platform != "linux", reason="ELF symbol isolation")
def test_rust_symbols_remain_local():
    # get_kernel has already loaded the extension. Its Rust export must not
    # become visible to unrelated extensions through the global symbol scope.
    with pytest.raises(AttributeError):
        getattr(ctypes.CDLL(None), "__tvm_ffi_relu_rust")
