"""Dataset-friendly benchmark result records.

A benchmark run is stored as a list of flat records, one per workload. Every
record carries the full run context (kernel, revision, machine, software
versions) so that records from many runs can be concatenated into a single
table, e.g. a Hugging Face dataset, and filtered or grouped without joins.

Records are written as JSON Lines with the fields declared in `FIELDS`. Use
`load_records` to load many files: `datasets.load_dataset("json", ...)` infers
the schema from the first file, which breaks on e.g. a column that is empty there.
Benchmark- and vendor-specific data goes into the `params` and `hardware` objects.
"""

import glob
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernels._internals.cli.benchmark import MachineInfo, TimingResults

# Bump when fields are renamed or change meaning. Adding fields is compatible.
SCHEMA_VERSION = 1

# Record fields and their types, in output order.
FIELDS: dict[str, str] = {
    "schema_version": "int",
    "run_id": "string",
    "timestamp": "string",
    # What was benchmarked
    "repo_id": "string",
    "revision": "string",
    "kernel_sha": "string",
    "benchmark_sha": "string",
    "benchmark": "string",
    "workload": "string",
    "params": "json",
    # Where it was benchmarked
    "gpu": "string",
    "gpu_cores": "int",
    "backend": "string",
    "cpu": "string",
    "os": "string",
    "torch_version": "string",
    "kernels_version": "string",
    "driver_version": "string",
    "gpu_max_clock_mhz": "int",
    "gpu_power_limit_w": "float",
    # How it was benchmarked
    "iterations": "int",
    "warmup": "int",
    "timing_mode": "string",
    # GPU state, sampled after the workload's timed loop
    "gpu_clock_mhz": "int",
    "gpu_mem_clock_mhz": "int",
    "gpu_power_draw_w": "float",
    "gpu_temperature_c": "int",
    "gpu_throttled": "bool",
    "hardware": "json",
    # Results
    "mean_ms": "float",
    "median_ms": "float",
    "std_ms": "float",
    "min_ms": "float",
    "max_ms": "float",
    "q1_ms": "float",
    "q3_ms": "float",
    "iqr_ms": "float",
    "outliers": "int",
    "verified": "bool",
    "ref_mean_ms": "float",
    "ref_median_ms": "float",
    "speedup": "float",
    "samples_ms": "list[float]",
    "ref_samples_ms": "list[float]",
}


def new_run_id() -> str:
    return uuid.uuid4().hex


def build_records(
    timing_results: "dict[str, TimingResults]",
    machine_info: "MachineInfo",
    *,
    repo_id: str,
    revision: str,
    kernel_sha: str,
    benchmark_sha: str | None,
    warmup: int,
    run_id: str | None = None,
    timestamp: datetime | None = None,
    kernels_version: str | None = None,
    include_samples: bool = True,
) -> list[dict[str, Any]]:
    """Flatten the results of one benchmark run into one record per workload."""
    if run_id is None:
        run_id = new_run_id()
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if kernels_version is None:
        kernels_version = _kernels_version()

    records = []
    for name, timing in sorted(timing_results.items()):
        benchmark, _, workload = name.rpartition(".")
        speedup = None
        if timing.ref_mean_ms is not None and timing.mean_ms > 0:
            speedup = round(timing.ref_mean_ms / timing.mean_ms, 4)

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": timestamp.isoformat(),
            # What was benchmarked
            "repo_id": repo_id,
            "revision": revision,
            "kernel_sha": kernel_sha,
            "benchmark_sha": benchmark_sha,
            "benchmark": benchmark,
            "workload": workload,
            "params": timing.params,
            # Where it was benchmarked
            "gpu": machine_info.gpu,
            "gpu_cores": machine_info.gpu_cores,
            "backend": machine_info.backend,
            "cpu": machine_info.cpu,
            "os": machine_info.os,
            "torch_version": machine_info.pytorch_version,
            "kernels_version": kernels_version,
            **(machine_info.gpu_config or {}),
            # How it was benchmarked
            "iterations": timing.iterations,
            "warmup": warmup,
            "timing_mode": timing.timing_mode,
            **(timing.gpu_state or {}),
            "hardware": {**(machine_info.hardware or {}), **(timing.hardware or {})} or None,
            # Results
            "mean_ms": timing.mean_ms,
            "median_ms": timing.median_ms,
            "std_ms": timing.std_ms,
            "min_ms": timing.min_ms,
            "max_ms": timing.max_ms,
            "q1_ms": timing.q1_ms,
            "q3_ms": timing.q3_ms,
            "iqr_ms": timing.iqr_ms,
            "outliers": timing.outliers,
            "verified": timing.verified,
            "ref_mean_ms": timing.ref_mean_ms,
            "ref_median_ms": timing.ref_median_ms,
            "speedup": speedup,
        }
        if include_samples:
            record["samples_ms"] = timing.samples_ms
            record["ref_samples_ms"] = timing.ref_samples_ms
        assert record.keys() <= FIELDS.keys(), f"undeclared record fields: {record.keys() - FIELDS.keys()}"
        records.append(normalize_record(record))
    return records


def write_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")


def read_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            try:
                records.append(normalize_record(record))
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from e
    return records


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a record to the current schema: missing fields become `None`, unknown fields are dropped."""
    version = record.get("schema_version")
    if version is None or version > SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark record schema version: {version}")
    return {name: _coerce(record.get(name), kind, name) for name, kind in FIELDS.items()}


def load_records(paths: "str | Path | list[str | Path]") -> Any:
    """Load records from JSON Lines files (paths or globs) as a `datasets.Dataset`, requires `datasets>=5`."""
    from datasets import (  # type: ignore[import-untyped,import-not-found,unused-ignore]
        Dataset,
        Features,
        Json,
        List,
        Value,
    )

    patterns = [paths] if isinstance(paths, (str, Path)) else paths
    files = sorted(f for p in patterns for f in glob.glob(str(p)))
    if not files:
        raise FileNotFoundError(f"No benchmark record files found: {paths!r}")
    types = {"string": Value("string"), "int": Value("int64"), "float": Value("float64"), "bool": Value("bool")}
    types |= {"list[float]": List(Value("float64")), "json": Json()}
    features = Features({name: types[kind] for name, kind in FIELDS.items()})
    return Dataset.from_list([r for f in files for r in read_records(f)], features=features)


def record_key(record: dict[str, Any]) -> str:
    """Key that identifies the same workload across runs."""
    return f"{record['benchmark']}.{record['workload']}"


def _coerce(value: Any, kind: str, name: str) -> Any:
    """Convert to the declared type, so that columns have the same type in every file."""
    if value is None:
        return None
    try:
        if kind == "json":
            if not isinstance(value, dict):
                raise TypeError(f"expected a dict, got {type(value).__name__}")
            # E.g. `torch.float16` becomes "float16".
            return json.loads(json.dumps(value, default=lambda v: str(v).removeprefix("torch.")))
        if kind == "list[float]":
            return [float(v) for v in value]
        return {"string": str, "int": int, "float": float, "bool": bool}[kind](value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid value for {name}: {value!r}: {e}") from e


def _kernels_version() -> str:
    from kernels import __version__

    return __version__
