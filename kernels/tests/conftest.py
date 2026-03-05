import sys

import pytest
import torch

from kernels.utils import _get_privateuse_backend_name

has_cuda = (
    hasattr(torch.version, "cuda")
    and torch.version.cuda is not None
    and torch.cuda.device_count() > 0
)

has_neuron = hasattr(torch, "neuron") and torch.neuron.device_count() > 0

has_rocm = (
    hasattr(torch.version, "hip")
    and torch.version.hip is not None
    and torch.cuda.device_count() > 0
)
has_xpu = (
    hasattr(torch.version, "xpu")
    and torch.version.xpu is not None
    and torch.xpu.device_count() > 0
)
has_npu = _get_privateuse_backend_name() == "npu"


def pytest_addoption(parser):
    parser.addoption(
        "--token",
        action="store_true",
        help="run tests that require a token with write permissions",
    )


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif _get_privateuse_backend_name() == "npu":
        return "npu"

    return "cpu"


def pytest_runtest_setup(item):
    if "cuda_only" in item.keywords and not has_cuda:
        pytest.skip("skipping CUDA-only test on host without CUDA")
    if "neuron_only" in item.keywords and not has_neuron:
        pytest.skip("skipping Neuron-only test on host without Neuron")
    if "rocm_only" in item.keywords and not has_rocm:
        pytest.skip("skipping ROCm-only test on host without ROCm")
    if "darwin_only" in item.keywords and not sys.platform.startswith("darwin"):
        pytest.skip("skipping macOS-only test on non-macOS platform")
    if "xpu_only" in item.keywords and not has_xpu:
        pytest.skip("skipping XPU-only test on host without XPU")
    if "npu_only" in item.keywords and not has_npu:
        pytest.skip("skipping NPU-only test on host without NPU")
    if "token" in item.keywords and not item.config.getoption("--token"):
        pytest.skip("need --token option to run this test")
