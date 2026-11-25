import kernels
import pytest

utils = kernels.get_kernel("kernels-test/sdk-arch-backend", version=1)


@pytest.mark.kernels_ci
def test_runtime_version():
    # This does not need a GPU.
    version = utils.runtime_version()
    assert 12000 <= version < 13000


@pytest.mark.kernels_ci
def test_device_count():
    assert utils.device_count() >= 0


@pytest.mark.kernels_ci
def test_cdiv():
    assert utils.cdiv(10, 3) == 4
    assert utils.cdiv(0, 3) == 0


@pytest.mark.kernels_ci
def test_round_up():
    assert utils.round_up(10, 8) == 16
    assert utils.round_up(16, 8) == 16


@pytest.mark.kernels_ci
def test_invalid_arguments():
    with pytest.raises(ValueError):
        utils.cdiv(1, 0)
    with pytest.raises(ValueError):
        utils.round_up(-1, 8)


@pytest.mark.kernels_ci
def test_cdiv_on_device():
    if utils.device_count() == 0:
        pytest.skip("No CUDA device")
    assert utils.cdiv_on_device(10, 3) == 4
