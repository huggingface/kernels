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
        monkeypatch.setattr(benchmark, "gpu_device", lambda backend_name: None)
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


def test_gpu_state_from_nvidia_smi(monkeypatch):
    from kernels._internals.cli import benchmark_gpu_state

    values = {
        "driver_version": "595.84",
        "clocks.max.sm": "3105",
        "power.limit": "450.00",
        "clocks.max.mem": "10501",
        "persistence_mode": "Disabled",
        "clocks.sm": "2805",
        "clocks.mem": "10251",
        "power.draw": "[N/A]",
        "temperature.gpu": "47",
        # Older drivers only know this name.
        "clocks_throttle_reasons.active": "0x0000000000000005",
    }

    def run(cmd, **kwargs):
        fields = cmd[2].removeprefix("--query-gpu=").split(",")
        if any(f not in values for f in fields):
            return SimpleNamespace(returncode=2, stdout="")
        return SimpleNamespace(returncode=0, stdout=", ".join(values[f] for f in fields))

    monkeypatch.setattr(benchmark_gpu_state.subprocess, "run", run)
    assert benchmark_gpu_state.gpu_config(("nvidia", "GPU-0")) == (
        {"driver_version": "595.84", "gpu_max_clock_mhz": 3105, "gpu_power_limit_w": 450.0},
        {"vendor": "nvidia", "max_mem_clock_mhz": 10501, "persistence_mode": "Disabled"},
    )
    assert benchmark_gpu_state.gpu_state(("nvidia", "GPU-0")) == (
        {
            "gpu_clock_mhz": 2805,
            "gpu_mem_clock_mhz": 10251,
            "gpu_power_draw_w": None,
            "gpu_temperature_c": 47,
            "gpu_throttled": True,
        },
        {"clock_event_reasons": ["gpu_idle", "sw_power_cap"]},
    )
    assert benchmark_gpu_state.gpu_state(None) == (dict.fromkeys(benchmark_gpu_state.STATE_FIELDS), {})


def test_gpu_state_from_amd_smi(monkeypatch):
    import json

    from kernels._internals.cli import benchmark_gpu_state

    # Trimmed `amd-smi --json` output of an MI300X.
    outputs = {
        "static": {"asic": {"target_graphics_version": "gfx942", "num_compute_units": 304}},
        "metric": {
            "clock": {
                "gfx_0": {
                    "clk": {"value": 2100, "unit": "MHz"},
                    "max_clk": {"value": 2100, "unit": "MHz"},
                    "clk_locked": "DISABLED",
                },
                "mem_0": {"clk": {"value": 1300, "unit": "MHz"}},
            },
            "power": {"socket_power": {"value": 450, "unit": "W"}, "throttle_status": "UNTHROTTLED"},
            "temperature": {"edge": "N/A", "hotspot": {"value": 60, "unit": "C"}},
        },
    }

    def run(cmd, **kwargs):
        assert cmd[2:4] == ["-g", "0000:83:00.0"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"gpu_data": [outputs[cmd[1]]]}))

    monkeypatch.setattr(benchmark_gpu_state.subprocess, "run", run)
    device = ("amd", "0000:83:00.0")
    assert benchmark_gpu_state.gpu_config(device) == (
        {"driver_version": None, "gpu_max_clock_mhz": 2100, "gpu_power_limit_w": None},
        {"vendor": "amd", "target_graphics_version": "gfx942", "num_compute_units": 304},
    )
    assert benchmark_gpu_state.gpu_state(device) == (
        {
            "gpu_clock_mhz": 2100,
            "gpu_mem_clock_mhz": 1300,
            "gpu_power_draw_w": 450,
            "gpu_temperature_c": 60,
            "gpu_throttled": False,
        },
        {"clk_locked": "DISABLED"},
    )


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
        self.x = torch.randn(64, 64, device=self.device)
        self.out = torch.empty_like(self.x)
        self.params = {"shape": tuple(self.x.shape), "dtype": self.x.dtype}

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
    monkeypatch.setattr(benchmark, "gpu_device", lambda backend_name: None)
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
    assert all(list(r) == list(benchmark_results.FIELDS) for r in records)
    assert small["params"] == {"shape": [64, 64], "dtype": "float32"}
    assert small["timing_mode"] == "eager"
    assert small["gpu_clock_mhz"] is None and small["hardware"] is None


def test_cuda_graph_falls_back_to_eager_on_cpu(fake_kernel_repo):
    script = (fake_kernel_repo / "benchmarks" / "benchmark_relu.py").read_text()
    (fake_kernel_repo / "benchmarks" / "benchmark_relu.py").write_text(
        script.replace("seed = 0", "seed = 0\n    cuda_graph = True")
    )
    result = benchmark.run_benchmark(str(fake_kernel_repo), branch=None, version=None, iterations=2, warmup=0)
    assert result.timing_results["ReluBenchmark.small"].timing_mode == "eager"


@pytest.mark.cuda_only
def test_cuda_graph_timing(fake_kernel_repo, monkeypatch):
    monkeypatch.setattr(benchmark, "_backend", lambda: CUDA(version=Version(torch.version.cuda)))
    script = (fake_kernel_repo / "benchmarks" / "benchmark_relu.py").read_text()
    (fake_kernel_repo / "benchmarks" / "benchmark_relu.py").write_text(
        script.replace("seed = 0", "seed = 0\n    cuda_graph = True")
    )
    result = benchmark.run_benchmark(str(fake_kernel_repo), branch=None, version=None, iterations=2, warmup=0)
    small = result.timing_results["ReluBenchmark.small"]
    assert small.timing_mode == "cuda_graph"
    assert small.verified is True


def test_read_records_rejects_newer_schema(tmp_path):
    import json

    from kernels._internals.cli import benchmark_results

    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"schema_version": benchmark_results.SCHEMA_VERSION + 1}) + "\n")
    with pytest.raises(ValueError, match="unsupported"):
        benchmark_results.read_records(path)


def test_normalize_record():
    from kernels._internals.cli.benchmark_results import FIELDS, normalize_record

    record = normalize_record({"schema_version": 1, "median_ms": 1, "from_a_newer_version": 1})
    assert list(record) == list(FIELDS)
    assert record["median_ms"] == 1.0 and isinstance(record["median_ms"], float)
    assert record["gpu_clock_mhz"] is None
    with pytest.raises(ValueError, match="params"):
        normalize_record({"schema_version": 1, "params": "small"})


def test_load_records_across_runs(tmp_path):
    """`datasets.load_dataset` cannot load these together, it infers the schema from the first file."""
    pytest.importorskip("datasets", minversion="5")
    from kernels._internals.cli.benchmark_results import load_records, write_records

    # No fields beyond the basics, e.g. written by an older version.
    (tmp_path / "a.jsonl").write_text('{"schema_version": 1, "benchmark": "B", "workload": "w", "median_ms": 1.0}\n')
    write_records(tmp_path / "b.jsonl", [_record("B", "w", 1.0, 0.9, 1.1, verified=True, params={"seq_len": 128})])
    write_records(
        tmp_path / "c.jsonl", [_record("B", "w", 1.0, 0.9, 1.1, hardware={"vendor": "amd", "gfx": "gfx942"})]
    )
    ds = load_records(str(tmp_path / "*.jsonl"))
    assert ds["params"] == [None, {"seq_len": 128}, None]
    assert ds["hardware"][2] == {"vendor": "amd", "gfx": "gfx942"}


def _record(benchmark_name, workload, median, q1, q3, **extra):
    from kernels._internals.cli.benchmark_results import SCHEMA_VERSION

    return {
        "schema_version": SCHEMA_VERSION,
        "repo_id": "kernels-community/relu",
        "revision": "v1",
        "kernel_sha": "abc1234",
        "benchmark": benchmark_name,
        "workload": workload,
        "gpu": "Fake GPU",
        "backend": "CUDA 12.8",
        "torch_version": "2.10.0",
        "median_ms": median,
        "q1_ms": q1,
        "q3_ms": q3,
        **extra,
    }


def test_compare_records_classifies_workloads():
    from kernels._internals.cli.benchmark_compare import Status, compare_records

    base = [
        _record("B", "faster", 10.0, 9.9, 10.1),
        _record("B", "noisy", 10.0, 5.0, 15.0),
        _record("B", "removed", 1.0, 0.9, 1.1),
        _record("B", "slower", 10.0, 9.9, 10.1),
        _record("B", "small_change", 10.0, 9.9, 10.1),
    ]
    new = [
        _record("B", "added", 1.0, 0.9, 1.1),
        _record("B", "faster", 5.0, 4.9, 5.1),
        _record("B", "noisy", 12.0, 6.0, 18.0),
        _record("B", "slower", 12.0, 11.9, 12.1),
        _record("B", "small_change", 10.2, 10.1, 10.3),
    ]
    comparison = compare_records(base, new, threshold=0.05)
    statuses = {w.key: w.status for w in comparison.workloads}
    assert statuses == {
        "B.added": Status.ADDED,
        "B.faster": Status.FASTER,
        "B.noisy": Status.UNCHANGED,
        "B.removed": Status.REMOVED,
        "B.slower": Status.SLOWER,
        "B.small_change": Status.UNCHANGED,
    }
    assert [w.key for w in comparison.regressions] == ["B.slower"]
    assert comparison.context_mismatches == {}


def test_compare_records_reports_context_mismatch():
    from kernels._internals.cli.benchmark_compare import compare_records

    base = [_record("B", "w", 1.0, 0.9, 1.1)]
    new = [_record("B", "w", 1.0, 0.9, 1.1, gpu="Other GPU")]
    assert compare_records(base, new).context_mismatches == {"gpu": ("Fake GPU", "Other GPU")}

    new = [_record("B", "w", 1.0, 0.9, 1.1, driver_version="595.84")]
    assert compare_records(base, new).context_mismatches == {"driver_version": (None, "595.84")}


def test_compare_records_notes():
    from kernels._internals.cli.benchmark_compare import compare_records

    base = [
        _record("B", "params", 1.0, 0.9, 1.1, params={"seq_len": 128}),
        _record("B", "power", 1.0, 0.9, 1.1, gpu_power_draw_w=100.0),
        _record("B", "noise", 1.0, 0.9, 1.1, gpu_power_draw_w=60.0),
    ]
    new = [
        _record("B", "params", 1.0, 0.9, 1.1, params={"seq_len": 4096}),
        _record("B", "power", 2.0, 1.9, 2.1, gpu_power_draw_w=380.0),
        # Unchanged timing, so the GPU state difference is not shown.
        _record("B", "noise", 1.0, 0.9, 1.1, gpu_power_draw_w=150.0),
    ]
    notes = {w.key: w.notes for w in compare_records(base, new).workloads}
    assert notes == {"B.noise": (), "B.params": ("params differ",), "B.power": ("power draw 100 vs 380 W",)}


def test_compare_fail_on_regression(tmp_path, capsys):
    from kernels._internals.cli.benchmark_compare import run_compare
    from kernels._internals.cli.benchmark_results import write_records

    base, new = tmp_path / "base.jsonl", tmp_path / "new.jsonl"
    write_records(base, [_record("B", "w", 1.0, 0.99, 1.01)])
    write_records(new, [_record("B", "w", 2.0, 1.99, 2.01)])

    run_compare(str(base), str(new))
    assert "1 slower" in capsys.readouterr().err

    with pytest.raises(SystemExit) as e:
        run_compare(str(base), str(new), fail_on_regression=True)
    assert e.value.code == 1


def test_run_benchmark_against_baseline(fake_kernel_repo, tmp_path, capsys):
    output = tmp_path / "base.jsonl"
    benchmark.run_benchmark(
        str(fake_kernel_repo), branch=None, version=None, iterations=5, warmup=1, output=str(output)
    )
    capsys.readouterr()
    benchmark.run_benchmark(
        str(fake_kernel_repo), branch=None, version=None, iterations=5, warmup=1, baseline=str(output)
    )
    err = capsys.readouterr().err
    assert "Base median(ms)" in err
    assert "ReluBenchmark.small" in err


def test_run_benchmark_downloads_benchmarks_from_kernel_repo(fake_kernel_repo, monkeypatch):
    import kernels

    calls = []

    class FakeApi:
        def snapshot_download(self, **kwargs):
            calls.append(kwargs)
            return str(fake_kernel_repo)

    monkeypatch.setattr(benchmark, "_get_hf_api", lambda: FakeApi())
    # Benchmarks call `kernel.relu`, so reuse the fake local kernel.
    monkeypatch.setattr(kernels, "get_kernel", lambda repo_id, revision: kernels.get_local_kernel(None))

    benchmark.run_benchmark("kernels-community/relu", branch=None, version=2, iterations=2, warmup=0)
    assert calls == [
        {
            "repo_id": "kernels-community/relu",
            "repo_type": "kernel",
            "revision": "v2",
            "allow_patterns": ["benchmarks/**"],
        }
    ]


@pytest.mark.parametrize("name", __import__("kernels._internals.benchmarks", fromlist=["__all__"]).__all__)
def test_builtin_benchmarks_use_runner_device(name):
    """Built-in benchmarks must allocate on the device chosen by the runner (#669)."""
    import kernels._internals.benchmarks

    cls = getattr(kernels._internals.benchmarks, name)
    setups = [m for m in dir(cls) if m.startswith("setup_")]
    assert setups
    for setup in setups:
        instance = cls()
        # The meta device allocates nothing, so even large workloads are cheap.
        instance.device = "meta"
        getattr(instance, setup)()
        tensors = [v for v in vars(instance).values() if isinstance(v, torch.Tensor)]
        assert tensors, f"{name}.{setup} did not create tensors"
        for t in tensors:
            if t.numel() > 1 or t.device.type != "cpu":
                assert t.device.type == "meta", f"{name}.{setup} allocated on {t.device}"


def test_builtin_benchmarks_are_public():
    import kernels._internals.benchmarks
    import kernels.benchmarks

    assert kernels.benchmarks.__all__ == kernels._internals.benchmarks.__all__
