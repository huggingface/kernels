from importlib.util import find_spec

import pytest

from kernels import get_kernel


@pytest.mark.cuda_only
@pytest.mark.parametrize("dependency", ["einops", "nvidia-cutlass-dsl"])
def test_python_deps(dependency):
    must_raise = find_spec(dependency.replace("-", "_")) is None
    if must_raise:
        with pytest.raises(
            ImportError,
            match=r"Kernel module `python_dep` requires Python dependency `(einops|nvidia-cutlass-dsl)`",
        ):
            get_kernel("kernels-test/python-dep", revision="main")
    else:
        get_kernel("kernels-test/python-dep", revision="main")


def test_illegal_dep():
    with pytest.raises(ValueError, match=r"Kernel module `python_invalid_dep` uses.*kepler-22b"):
        get_kernel("kernels-test/python-invalid-dep", revision="main")
