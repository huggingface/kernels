"""Compare two benchmark runs to detect performance changes.

Comparisons use the median wall time, which is robust against the occasional slow iteration.
A workload only counts as changed when the median moved by more than the threshold *and* the
interquartile ranges of both runs do not overlap, so that noisy workloads are not reported as regressions.
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kernels._internals.cli.benchmark_results import record_key

DEFAULT_THRESHOLD = 0.05

# Fields that should match for two runs to be directly comparable.
_CONTEXT_FIELDS = ("gpu", "backend", "torch_version", "driver_version", "gpu_power_limit_w")
# Relative differences of the sampled GPU state that are worth a note.
CLOCK_DIFF_THRESHOLD = 0.1
POWER_DIFF_THRESHOLD = 0.5


class Status(str, Enum):
    FASTER = "faster"
    SLOWER = "slower"
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"


@dataclass
class WorkloadComparison:
    key: str
    status: Status
    base_ms: float | None = None
    new_ms: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def change(self) -> float | None:
        """Relative change of the median, positive means slower."""
        if self.base_ms is None or self.new_ms is None or self.base_ms <= 0:
            return None
        return self.new_ms / self.base_ms - 1.0


@dataclass
class Comparison:
    workloads: list[WorkloadComparison]
    context_mismatches: dict[str, tuple[Any, Any]]
    threshold: float

    @property
    def regressions(self) -> list[WorkloadComparison]:
        return [w for w in self.workloads if w.status == Status.SLOWER]


def compare_records(
    base: list[dict[str, Any]],
    new: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> Comparison:
    base_by_key = {record_key(r): r for r in base}
    new_by_key = {record_key(r): r for r in new}

    workloads = []
    for key in sorted(base_by_key.keys() | new_by_key.keys()):
        b = base_by_key.get(key)
        n = new_by_key.get(key)
        if b is None:
            assert n is not None
            workloads.append(WorkloadComparison(key, Status.ADDED, new_ms=n["wall_median_ms"]))
        elif n is None:
            workloads.append(WorkloadComparison(key, Status.REMOVED, base_ms=b["wall_median_ms"]))
        else:
            status = _classify(b, n, threshold)
            workloads.append(
                WorkloadComparison(
                    key, status, b["wall_median_ms"], n["wall_median_ms"], notes=_condition_notes(b, n, status)
                )
            )

    return Comparison(
        workloads=workloads,
        context_mismatches=_context_mismatches(base, new),
        threshold=threshold,
    )


def _classify(base: dict[str, Any], new: dict[str, Any], threshold: float) -> Status:
    base_ms, new_ms = base["wall_median_ms"], new["wall_median_ms"]
    if base_ms <= 0:
        return Status.UNCHANGED
    change = new_ms / base_ms - 1.0
    if abs(change) <= threshold:
        return Status.UNCHANGED
    if change > 0 and new["wall_q1_ms"] > base["wall_q3_ms"]:
        return Status.SLOWER
    if change < 0 and new["wall_q3_ms"] < base["wall_q1_ms"]:
        return Status.FASTER
    # Median moved, but the distributions overlap: within noise.
    return Status.UNCHANGED


def _condition_notes(base: dict[str, Any], new: dict[str, Any], status: Status) -> tuple[str, ...]:
    notes = []
    if base.get("timing_mode") != new.get("timing_mode"):
        notes.append(f"timing mode {base.get('timing_mode')} vs {new.get('timing_mode')}")
    if base.get("params") != new.get("params"):
        notes.append("params differ")
    # A single sample of the GPU state is noisy, only use it to explain a change.
    if status == Status.UNCHANGED:
        return tuple(notes)
    base_clock, new_clock = base.get("gpu_clock_mhz"), new.get("gpu_clock_mhz")
    if base_clock and new_clock and abs(new_clock / base_clock - 1.0) > CLOCK_DIFF_THRESHOLD:
        notes.append(f"clock {base_clock} vs {new_clock} MHz")
    base_power, new_power = base.get("gpu_power_draw_w"), new.get("gpu_power_draw_w")
    if base_power and new_power and abs(new_power / base_power - 1.0) > POWER_DIFF_THRESHOLD:
        notes.append(f"power draw {base_power:.0f} vs {new_power:.0f} W")
    return tuple(notes)


def _context_mismatches(base: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
    if not base or not new:
        return {}
    mismatches = {}
    for field in _CONTEXT_FIELDS:
        b, n = base[0].get(field), new[0].get(field)
        if b != n:
            mismatches[field] = (b, n)
    return mismatches


def _run_label(records: list[dict[str, Any]]) -> str:
    if not records:
        return "(empty)"
    r = records[0]
    return f"{r['repo_id']}@{r['revision']} ({r['kernel_sha'][:7]})"


def print_comparison(
    comparison: Comparison,
    base: list[dict[str, Any]],
    new: list[dict[str, Any]],
    file=None,
) -> None:
    from tabulate import tabulate

    if file is None:
        file = sys.stderr

    print(file=file)
    print(f"  Base  {_run_label(base)}", file=file)
    print(f"  New   {_run_label(new)}", file=file)
    for field, (b, n) in comparison.context_mismatches.items():
        print(f"  Warning: {field} differs between runs: {b!r} vs {n!r}", file=file)

    symbols = {
        Status.FASTER: "▲ faster",
        Status.SLOWER: "▼ slower",
        Status.UNCHANGED: "· same",
        Status.ADDED: "+ added",
        Status.REMOVED: "- removed",
    }
    rows = []
    for w in comparison.workloads:
        change = f"{w.change * 100:+.1f}%" if w.change is not None else ""
        rows.append(
            [
                w.key,
                f"{w.base_ms:.4f}" if w.base_ms is not None else "",
                f"{w.new_ms:.4f}" if w.new_ms is not None else "",
                change,
                symbols[w.status],
            ]
        )
    print(file=file)
    print(
        tabulate(
            rows,
            headers=["Workload", "Base median(ms)", "New median(ms)", "Change", "Status"],
            tablefmt="simple_outline",
        ),
        file=file,
    )

    for w in comparison.workloads:
        if w.notes:
            print(f"  Note: {w.key}: {', '.join(w.notes)}", file=file)

    n_regressions = len(comparison.regressions)
    n_faster = sum(1 for w in comparison.workloads if w.status == Status.FASTER)
    print(file=file)
    print(
        f"  {n_faster} faster, {n_regressions} slower (threshold {comparison.threshold * 100:.0f}%)",
        file=file,
    )
    print(file=file)


def run_compare(
    base_path: str,
    new_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    fail_on_regression: bool = False,
) -> Comparison:
    """Compare two saved runs and print the result."""
    from kernels._internals.cli.benchmark_results import read_records

    base = read_records(base_path)
    new = read_records(new_path)
    return report(base, new, threshold=threshold, fail_on_regression=fail_on_regression)


def report(
    base: list[dict[str, Any]],
    new: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    fail_on_regression: bool = False,
) -> Comparison:
    comparison = compare_records(base, new, threshold=threshold)
    print_comparison(comparison, base, new)
    if fail_on_regression and comparison.regressions:
        keys = ", ".join(w.key for w in comparison.regressions)
        print(f"Error: performance regression in: {keys}", file=sys.stderr)
        sys.exit(1)
    return comparison
