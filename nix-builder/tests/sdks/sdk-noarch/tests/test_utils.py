import kernels
import pytest

utils = kernels.get_kernel("kernels-test/sdk-noarch", version=1)


@pytest.mark.kernels_ci
def test_cdiv():
    assert utils.cdiv(10, 3) == 4
    assert utils.cdiv(9, 3) == 3
    assert utils.cdiv(0, 3) == 0


@pytest.mark.kernels_ci
def test_round_up():
    assert utils.round_up(10, 8) == 16
    assert utils.round_up(16, 8) == 16
    assert utils.round_up(0, 8) == 0


@pytest.mark.kernels_ci
def test_invalid_arguments():
    with pytest.raises(ValueError):
        utils.cdiv(1, 0)
    with pytest.raises(ValueError):
        utils.round_up(-1, 8)
