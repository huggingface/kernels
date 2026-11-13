import kernels
import mlx.core as mx
import pytest

relu = kernels.get_kernel("kernels-test/relu-mlx", version=1)


@pytest.mark.kernels_ci
def test_relu():
    x = mx.random.normal((1024, 1024))
    assert mx.array_equal(relu.relu(x), mx.maximum(x, 0))
