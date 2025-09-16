import sys

import pytest
import torch

has_cuda = (
    hasattr(torch.version, "cuda")
    and torch.version.cuda is not None
    and torch.cuda.device_count() > 0
)
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


def pytest_addoption(parser):
    parser.addoption(
        "--token",
        action="store_true",
        help="run tests that require a token with write permissions",
    )


def pytest_runtest_setup(item):
    if "cuda_only" in item.keywords and not has_cuda:
        pytest.skip("skipping CUDA-only test on host without CUDA")
    if "rocm_only" in item.keywords and not has_rocm:
        pytest.skip("skipping ROCm-only test on host without ROCm")
    if "darwin_only" in item.keywords and not sys.platform.startswith("darwin"):
        pytest.skip("skipping macOS-only test on non-macOS platform")
    if "xpu_only" in item.keywords and not has_xpu:
        pytest.skip("skipping XPU-only test on host without XPU")
    if "token" in item.keywords and not item.config.getoption("--token"):
        pytest.skip("need --token option to run this test")
