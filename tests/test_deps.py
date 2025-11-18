from importlib.util import find_spec

import pytest

from kernels import get_kernel


def test_python_deps():
    must_raise = find_spec("nvidia_cutlass_dsl") is None
    if must_raise:
        with pytest.raises(
            ImportError, match=r"Kernel requires dependency `nvidia-cutlass-dsl`"
        ):
            get_kernel("kernels-test/python-dep")
    else:
        get_kernel("kernels-test/python-dep")


def test_illegal_dep():
    with pytest.raises(ValueError, match=r"Invalid dependency: kepler-22b"):
        get_kernel("kernels-test/python-invalid-dep")
