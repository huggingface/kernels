import sys

import pytest
import torch

has_cuda = hasattr(torch.version, 'cuda') and torch.version.cuda is not None and torch.cuda.device_count() > 0
has_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.device_count() > 0


def pytest_runtest_setup(item):
    if "cuda_only" in item.keywords and not has_cuda:
        pytest.skip("skipping CUDA-only test on host without CUDA")
    if "rocm_only" in item.keywords and not has_rocm:
        pytest.skip("skipping ROCm-only test on host without ROCm")
    if "darwin_only" in item.keywords and not sys.platform.startswith("darwin"):
        pytest.skip("skipping macOS-only test on non-macOS platform")
