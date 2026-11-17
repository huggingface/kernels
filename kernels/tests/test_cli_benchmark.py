from types import SimpleNamespace

import pytest
from packaging.version import Version

from kernels._internals.backends import CPU, CUDA, XPU, Backend, ROCm

torch = pytest.importorskip("torch")

from kernels._internals.cli import benchmark  # noqa: E402


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


def test_timing_results_stats():
    t = benchmark._timing_results([1.0, 2.0, 3.0, 4.0, 100.0], ref_times_ms=[10.0, 20.0, 30.0])
    assert t.iterations == 5
    assert t.mean_ms == 22.0
    assert t.median_ms == 3.0
    assert t.min_ms == 1.0
    assert t.max_ms == 100.0
    assert t.outliers == 1
    assert t.ref_mean_ms == 20.0
    assert t.ref_median_ms == 20.0
    assert t.samples_ms == [1.0, 2.0, 3.0, 4.0, 100.0]
    assert t.ref_samples_ms == [10.0, 20.0, 30.0]


def test_time_fn_runs_warmup_and_iterations():
    calls = []
    samples = benchmark._time_fn(lambda: calls.append(1), iterations=7, warmup=3)
    assert len(calls) == 10
    assert len(samples) == 7
    assert all(s >= 0 for s in samples)


FAKE_BENCHMARK_SCRIPT = """
import torch

from kernels._internals.benchmark import Benchmark


class ReluBenchmark(Benchmark):
    seed = 0

    def setup(self):
        self.x = torch.randn(64, 64)
        self.out = torch.empty_like(self.x)

    def benchmark_small(self):
        self.kernel.relu(self.out, self.x)

    def verify_small(self):
        return torch.relu(self.x)

    def benchmark_unverified(self):
        self.kernel.relu(self.out, self.x)
"""


@pytest.fixture
def fake_kernel_repo(tmp_path, monkeypatch):
    """A local kernel repo with a benchmark script and a fake loaded kernel."""
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "benchmark_relu.py").write_text(FAKE_BENCHMARK_SCRIPT)

    def relu(out, x):
        out.copy_(torch.relu(x))

    fake_kernel = SimpleNamespace(__name__="relu_abc1234", relu=relu)
    import kernels

    monkeypatch.setattr(kernels, "get_local_kernel", lambda path: fake_kernel)
    monkeypatch.setattr(benchmark, "_backend", lambda: CPU())
    return tmp_path


def test_run_benchmark_local(fake_kernel_repo):
    result = benchmark.run_benchmark(str(fake_kernel_repo), branch=None, version=None, iterations=5, warmup=1)
    assert result.kernel_commit_sha == "abc1234"

    small = result.timing_results["ReluBenchmark.small"]
    assert small.verified is True
    assert small.iterations == 5
    assert len(small.samples_ms) == 5
    assert small.ref_mean_ms is not None
    assert len(small.ref_samples_ms) == 5

    unverified = result.timing_results["ReluBenchmark.unverified"]
    assert unverified.verified is None
    assert unverified.ref_mean_ms is None


def test_run_benchmark_writes_records(fake_kernel_repo, tmp_path):
    from kernels._internals.cli import benchmark_results

    output = tmp_path / "results.jsonl"
    benchmark.run_benchmark(
        str(fake_kernel_repo), branch=None, version=None, iterations=5, warmup=1, output=str(output)
    )

    records = benchmark_results.read_records(output)
    assert [(r["benchmark"], r["workload"]) for r in records] == [
        ("ReluBenchmark", "small"),
        ("ReluBenchmark", "unverified"),
    ]
    assert len({r["run_id"] for r in records}) == 1
    small, unverified = records
    assert small["schema_version"] == benchmark_results.SCHEMA_VERSION
    assert small["revision"] == "local"
    assert small["kernel_sha"] == "abc1234"
    assert small["backend"] == "CPU"
    assert small["iterations"] == 5
    assert small["warmup"] == 1
    assert small["verified"] is True
    assert small["speedup"] is not None
    assert len(small["samples_ms"]) == 5
    assert unverified["verified"] is None
    assert unverified["speedup"] is None
    assert unverified["ref_samples_ms"] is None


def test_read_records_rejects_newer_schema(tmp_path):
    import json

    from kernels._internals.cli import benchmark_results

    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"schema_version": benchmark_results.SCHEMA_VERSION + 1}) + "\n")
    with pytest.raises(ValueError, match="unsupported"):
        benchmark_results.read_records(path)
