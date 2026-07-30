from types import ModuleType
from typing import Tuple

from kernels import get_kernel
import pytest
import torch


@pytest.fixture
def kernels() -> Tuple[ModuleType, ModuleType]:
    # These are not actually uploaded kernels, but we need to use
    # a trusted namespace to avoid attacks.
    kernel = get_kernel("kernels-test/symbol-conflicts", version=1)
    kernel2 = get_kernel("kernels-test/symbol-conflicts2", version=1)
    return kernel, kernel2


@pytest.mark.kernels_ci
def test_conflicts(kernels):
    kernel, kernel2 = kernels
    scalar = torch.Tensor(10)
    kernel.conflicts(scalar)
    kernel.conflicts(scalar)
    assert kernel.conflicts(scalar) == 2
    assert kernel2.conflicts(scalar) == 0
    assert kernel2.conflicts(scalar) == 1


@pytest.mark.kernels_ci
def test_conflicts_dynamic(kernels):
    kernel, kernel2 = kernels
    scalar = torch.Tensor(10)
    kernel.conflicts_dynamic(scalar)
    kernel.conflicts_dynamic(scalar)
    assert kernel.conflicts_dynamic(scalar) == 2
    assert kernel2.conflicts_dynamic(scalar) == 0
    assert kernel2.conflicts_dynamic(scalar) == 1


@pytest.mark.kernels_ci
def test_conflicts_class(kernels):
    kernel, kernel2 = kernels
    scalar = torch.Tensor(10)
    kernel.conflicts_class(scalar)
    kernel.conflicts_class(scalar)
    assert kernel.conflicts_class(scalar) == 2
    assert kernel2.conflicts_class(scalar) == 0
    assert kernel2.conflicts_class(scalar) == 1
