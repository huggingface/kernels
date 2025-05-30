import sys

import pytest


def pytest_runtest_setup(item):
    if "linux_only" in item.keywords and not sys.platform.startswith("linux"):
        pytest.skip("skipping Linux-only test on non-Linux platform")
    if "darwin_only" in item.keywords and not sys.platform.startswith("darwin"):
        pytest.skip("skipping macOS-only test on non-macOS platform")
