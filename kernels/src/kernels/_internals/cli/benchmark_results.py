"""Dataset-friendly benchmark result records.

A benchmark run is stored as a list of flat records, one per workload. Every
record carries the full run context (kernel, revision, machine, software
versions) so that records from many runs can be concatenated into a single
table, e.g. a Hugging Face dataset, and filtered or grouped without joins.

Records are written as JSON Lines, which `datasets.load_dataset("json", ...)`
reads directly.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kernels._internals.cli.benchmark import MachineInfo, TimingResults

# Bump when fields are renamed or change meaning. Adding fields is compatible.
SCHEMA_VERSION = 1


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
            # Where it was benchmarked
            "gpu": machine_info.gpu,
            "gpu_cores": machine_info.gpu_cores,
            "backend": machine_info.backend,
            "cpu": machine_info.cpu,
            "os": machine_info.os,
            "torch_version": machine_info.pytorch_version,
            "kernels_version": kernels_version,
            # How it was benchmarked
            "iterations": timing.iterations,
            "warmup": warmup,
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
        records.append(record)
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
            version = record.get("schema_version")
            if version is None or version > SCHEMA_VERSION:
                raise ValueError(f"{path}:{lineno}: unsupported benchmark record schema version: {version}")
            records.append(record)
    return records


def record_key(record: dict[str, Any]) -> str:
    """Key that identifies the same workload across runs."""
    return f"{record['benchmark']}.{record['workload']}"


def _kernels_version() -> str:
    from kernels import __version__

    return __version__
