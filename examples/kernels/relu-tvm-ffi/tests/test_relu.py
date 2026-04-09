import pytest
import torch
import torch.nn.functional as F

import relu_tvm_ffi


@pytest.mark.kernels_ci
def test_relu(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    torch.testing.assert_close(F.relu(x), relu_tvm_ffi.relu(x, torch.empty_like(x)))


@pytest.mark.kernels_ci
def test_relu_views(device):
    x = torch.arange(-20, 20, device=device, dtype=torch.float32)

    # Keep buffers and fill on each iteration. Stable pointers make C++-side
    # pointer inspection easier.
    out = torch.empty_like(x)
    out_check = torch.empty_like(x)

    for i in range(41):
        # Put a sentineal value in the output.
        out.fill_(42)
        out_check.fill_(42)

        relu_tvm_ffi.relu(x[i:], out[i:])
        out_check[i:] = F.relu(x[i:])

        torch.testing.assert_close(out, out_check)


@pytest.mark.kernels_ci
def test_relu_layer(device):
    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    layer = relu_tvm_ffi.layers.ReLU()
    torch.testing.assert_close(F.relu(x), layer(x))


def test_numpy(device):
    if device.type not in ["cpu"]:
        pytest.skip("skipping NumPy test on non-CPU device")

    import numpy as np

    x = np.arange(-20, 20, dtype=np.float32)
    # Keep buffers and fill on each iteration. Stable pointers make C++-side
    # pointer inspection easier.
    out = np.empty_like(x)
    out_check = np.empty_like(x)

    for i in range(41):
        # Put a sentineal value in the output.
        out.fill(42)
        out_check.fill(42)

        relu_tvm_ffi.relu(x[i:], out[i:])
        out_check[i:] = np.maximum(x[i:], 0)

        np.testing.assert_allclose(out, out_check)


@pytest.mark.jax_only
def test_relu_jax(device):
    import jax
    import jax.numpy as jnp
    from numpy.testing import assert_allclose

    if device.type == "cpu":
        device = jax.devices("cpu")[0]
    else:
        device = jax.devices()[0]

    x = jax.device_put(jnp.arange(-20, 20, dtype=jnp.float32), device)

    for i in range(41):
        out = relu_tvm_ffi.relu_jax(x[i:])
        expected = jax.nn.relu(x[i:])
        assert_allclose(out, expected)
