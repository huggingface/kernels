import importlib.util
import os
import subprocess
import sys

import pytest

from kernels.backends import _get_torch_privateuse_backend_name

try:
    import torch
except ImportError:
    torch = None

has_cuda = (
    torch is not None
    and hasattr(torch.version, "cuda")
    and torch.version.cuda is not None
    and torch.cuda.device_count() > 0
)

has_neuron = (
    torch is not None and hasattr(torch, "neuron") and torch.neuron.device_count() > 0
)

has_rocm = (
    torch is not None
    and hasattr(torch.version, "hip")
    and torch.version.hip is not None
    and torch.cuda.device_count() > 0
)
has_xpu = (
    torch is not None
    and hasattr(torch.version, "xpu")
    and torch.version.xpu is not None
    and torch.xpu.device_count() > 0
)

has_npu = torch is not None and _get_torch_privateuse_backend_name() == "npu"

has_jax = (
    importlib.util.find_spec("jax") is not None
    and importlib.util.find_spec("jax_tvm_ffi") is not None
)


def pytest_addoption(parser):
    parser.addoption(
        "--token",
        action="store_true",
        help="run tests that require a token with write permissions",
    )


@pytest.fixture
def device():
    if has_cuda:
        return "cuda"
    elif has_xpu:
        return "xpu"
    elif has_npu:
        return "npu"

    return "cpu"


def pytest_runtest_setup(item):
    if "torch_only" in item.keywords and torch is None:
        pytest.skip("skipping CUDA Torch-only test on host without Torch")
    if "cuda_only" in item.keywords and not has_cuda:
        pytest.skip("skipping CUDA-only test on host without CUDA")
    if "jax_only" in item.keywords and not has_jax:
        pytest.skip("skipping JAX-only test on host without JAX")
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
