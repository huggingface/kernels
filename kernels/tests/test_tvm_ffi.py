import pytest

from kernels import get_kernel, get_local_kernel, install_kernel

relu_supported_devices = ["cpu", "cuda", "xpu"]


def test_relu_load(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")
    get_kernel("kernels-test/relu-tvm-ffi", version=1)


@pytest.mark.torch_only
def test_relu_torch(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")
    kernel = get_kernel("kernels-test/relu-tvm-ffi", version=1)

    import torch

    x = torch.arange(-10, 10, dtype=torch.float32, device=device)
    out = kernel.relu(x, torch.empty_like(x))

    torch.testing.assert_close(out, torch.relu(x))


def test_local_load(device):
    if device not in relu_supported_devices:
        pytest.skip(f"Device is not one of: {','.join(relu_supported_devices)}")

    package_name, path = install_kernel("kernels-test/relu-tvm-ffi", "v1")
    get_local_kernel(path.parent.parent, package_name)


@pytest.mark.cuda_only
@pytest.mark.jax_only
def test_jax(device):
    import jax
    import jax_tvm_ffi
    import numpy as np

    kernel = get_kernel("kernels-test/relu-tvm-ffi", version=1)

    x = jax.numpy.arange(-10, 10, dtype=jax.numpy.float32)
    jax_tvm_ffi.register_ffi_target("relu.relu", kernel.relu, platform="gpu")
    out = jax.ffi.ffi_call(
        "relu.relu", jax.ShapeDtypeStruct(x.shape, x.dtype), vmap_method="broadcast_all"
    )(x)

    expected = jax.nn.relu(x)
    np.testing.assert_array_equal(out, expected)
