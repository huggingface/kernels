import sys

import pytest
import torch

has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0


def pytest_runtest_setup(item):
    if "cuda_only" in item.keywords and not has_cuda:
        pytest.skip("skipping CUDA-only test on host without CUDA")
    if "darwin_only" in item.keywords and not sys.platform.startswith("darwin"):
        pytest.skip("skipping macOS-only test on non-macOS platform")
