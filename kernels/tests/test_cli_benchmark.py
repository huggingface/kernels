from types import SimpleNamespace

import pytest
from packaging.version import Version

from kernels.backends import CPU, CUDA, XPU, Backend, ROCm

torch = pytest.importorskip("torch")

from kernels.cli import benchmark  # noqa: E402


@pytest.fixture
def machine_info(monkeypatch):
    """Run `collect_machine_info` with a fake backend and fake devices."""

    def collect(backend: Backend, hip: str | None = None) -> benchmark.MachineInfo:
        monkeypatch.setattr(benchmark, "_backend", lambda: backend)
        monkeypatch.setattr(benchmark.platform, "system", lambda: "Linux")
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake CUDA GPU")
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(get_device_name=lambda index: "Fake XPU GPU"), raising=False)
        monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
        monkeypatch.setattr(torch.version, "hip", hip, raising=False)
        monkeypatch.setattr(torch.version, "rocm", hip, raising=False)
        return benchmark.collect_machine_info()

    return collect


def test_machine_info_cuda(machine_info):
    info = machine_info(CUDA(version=Version("12.8")))
    assert info.gpu == "Fake CUDA GPU"
    assert info.backend == "CUDA 12.8"


def test_machine_info_rocm(machine_info):
    info = machine_info(ROCm(version=Version("6.4")), hip="6.4.0")
    assert info.gpu == "Fake CUDA GPU"
    assert info.backend == "ROCm 6.4.0"


def test_machine_info_xpu(machine_info):
    info = machine_info(XPU(version=Version("2025.3")))
    assert info.gpu == "Fake XPU GPU"
    assert info.backend == "XPU"


def test_machine_info_cpu(machine_info):
    info = machine_info(CPU())
    assert info.gpu == "N/A"
    assert info.backend == "CPU"
