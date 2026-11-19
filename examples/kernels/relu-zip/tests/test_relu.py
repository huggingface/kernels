import importlib.resources

import kernels
import pytest
import torch
import torch.nn.functional as F

relu = kernels.get_kernel("kernels-test/relu-zip", version=1)


@pytest.mark.kernels_ci
def test_relu(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_views(device):
    x = torch.arange(-20, 20, device=device, dtype=torch.float32)

    # Keep buffers and fill on each iteration. Stable pointers make C++-side
    # pointer inspection esier.
    out = torch.empty_like(x)
    out_check = torch.empty_like(x)

    for i in range(41):
        # Put a sentineal value in the output.
        out.fill_(42)
        out_check.fill_(42)

        relu.relu(x[i:], out[i:])
        out_check[i:] = F.relu(x[i:])

        torch.testing.assert_close(out, out_check)


@pytest.mark.kernels_ci
def test_relu_layer(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_relu_triton(device, dtype):
    if device.type not in ("cuda", "xpu"):
        pytest.skip("Triton implementation requires a CUDA, ROCm, or XPU device")

    x = torch.randn(1024, 1024, dtype=dtype, device=device)
    torch.testing.assert_close(F.relu(x), relu.relu_triton(x))


@pytest.mark.kernels_ci
def test_tuned_configs_are_packaged():
    # Data files must be available, also when the kernel is stored in a ZIP archive.
    configs = importlib.resources.files(relu.__name__) / "configs"
    assert any(path.name.endswith(".json") for path in configs.iterdir())


@pytest.mark.kernels_ci
def test_relu_triton_uses_tuned_config(device):
    if device.type != "cuda" or torch.cuda.get_device_name(device) != "NVIDIA L4":
        pytest.skip("Tuned configurations are only available for the NVIDIA L4")

    x = torch.empty(2**20, dtype=torch.float32, device=device)
    # The default configuration is `BLOCK_SIZE=1024, num_warps=4`.
    assert relu.triton_relu.select_config(x) == {"BLOCK_SIZE": 1024, "num_warps": 1}
