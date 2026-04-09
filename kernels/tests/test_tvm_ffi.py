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

    package_name, path = install_kernel("kernels-test/relu-tvm-ffi", "v1", repo_type="model")
    get_local_kernel(path.parent.parent, package_name)
