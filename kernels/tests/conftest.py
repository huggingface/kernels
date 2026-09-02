import importlib.util
import json
import sys
from pathlib import Path

import pytest
from kernels_data import Metadata

from kernels.backends import _get_torch_privateuse_backend_name
from kernels.deps import DepTreeNode
from kernels.resolver import LocalKernel

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

has_neuron = torch is not None and hasattr(torch, "neuron") and torch.neuron.device_count() > 0

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

has_tpu = torch is not None and hasattr(torch, "tpu") and torch.tpu.device_count() > 0

has_jax = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("jax_tvm_ffi") is not None


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
    elif has_tpu:
        return "tpu"

    return "cpu"


@pytest.fixture
def make_metadata():
    def _make_metadata(backend_type: str, archs: list[str] | None, kernels_minver: str | None = None) -> Metadata:
        metadata = {
            "name": "test-kernel",
            "id": "test_id",
            "version": 1,
            "license": "mit",
            "python-depends": [],
            "backend": {"type": backend_type, "archs": archs},
        }
        if kernels_minver is not None:
            metadata["kernels-minver"] = kernels_minver
        return Metadata.from_bytes(json.dumps(metadata).encode("utf-8"))

    return _make_metadata


@pytest.fixture
def make_tree():
    def _make_tree(metadata: Metadata, variant: str = "test-variant") -> DepTreeNode:
        return DepTreeNode(
            location=LocalKernel(Path(variant), metadata),
            deps={},
        )

    return _make_tree


@pytest.fixture
def fake_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))


@pytest.fixture
def fake_rocm_device(monkeypatch):
    class FakeProperties:
        gcnArchName = "gfx90a:sramecc+:xnack-"

    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: FakeProperties())


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
    if "tpu_only" in item.keywords and not has_tpu:
        pytest.skip("skipping TPU-only test on host without TPU")
    if "token" in item.keywords and not item.config.getoption("--token"):
        pytest.skip("need --token option to run this test")
