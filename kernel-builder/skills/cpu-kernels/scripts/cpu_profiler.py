#!/usr/bin/env python3
"""
Profile a CPU kernel using perf stat hardware counters.

Collects IPC, cache misses, branch mispredictions and maps bottlenecks to
optimization patterns. Use when speedup plateaus or you need guidance on
which optimization to try next.

Usage:
    python scripts/cpu_profiler.py --kernel-package my_kernel --op my_kernel.forward
    python scripts/cpu_profiler.py --kernel-package my_kernel --op my_kernel.forward --warmup 10 --iters 50
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config as _load_config

_CFG = _load_config()


def _find_perf_binary():
    """Find a working perf binary, handling kernel version mismatches."""
    import glob

    # Try the standard perf wrapper first
    try:
        result = subprocess.run(
            ["perf", "stat", "echo", "test"],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            return "perf"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: look for versioned perf binaries directly
    perf_candidates = sorted(glob.glob("/usr/lib/linux-tools/*/perf"), reverse=True)
    for candidate in perf_candidates:
        try:
            result = subprocess.run(
                [candidate, "stat", "echo", "test"],
                capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def _check_perf_available():
    """Check if perf stat is available."""
    return _find_perf_binary() is not None


def generate_runner_script(kernel_package: str, op_path: str, warmup: int, iters: int, baseline_file: str = None) -> str:
    """Generate a Python script that runs the kernel for profiling."""
    baseline_path = baseline_file or "baseline.py"
    return f"""\
import importlib
import torch

# Load kernel
parts = "{op_path}".split(".")
mod = importlib.import_module(parts[0])
func = mod
for attr in parts[1:]:
    func = getattr(func, attr)

# Create sample inputs (adjust as needed)
# For profiling, we use representative shapes
torch.set_num_threads(torch.get_num_threads())

# Try to load from baseline if available
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("baseline", "{baseline_path}")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    inputs = baseline.get_inputs()
except Exception:
    print("Warning: Could not load baseline for inputs. Using dummy inputs.")
    inputs = [torch.randn(1024, 4096, dtype=torch.bfloat16)]

# Warmup
for _ in range({warmup}):
    func(*inputs)

# Profiled iterations
for _ in range({iters}):
    func(*inputs)
"""


def run_perf_stat(kernel_package: str, op_path: str, warmup: int, iters: int, baseline_file: str = None) -> dict:
    """Run perf stat and parse results."""
    script = generate_runner_script(kernel_package, op_path, warmup, iters, baseline_file)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        # perf stat counters for CPU kernel analysis
        counters = [
            "task-clock",
            "cycles",
            "instructions",
            "cache-references",
            "cache-misses",
            "L1-dcache-loads",
            "L1-dcache-load-misses",
            "LLC-loads",
            "LLC-load-misses",
            "branch-instructions",
            "branch-misses",
            "page-faults",
        ]

        perf_bin = _find_perf_binary()
        if perf_bin is None:
            print("  perf binary not found")
            return {}

        cmd = [
            perf_bin, "stat",
            "-e", ",".join(counters),
            "--", sys.executable, script_path,
        ]

        print(f"  Running: {' '.join(cmd[:6])} ...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"  perf stat failed (exit code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return {}

        # Parse perf stat output (from stderr)
        stats = _parse_perf_output(result.stderr)
        return stats

    finally:
        os.unlink(script_path)


def _parse_perf_output(output: str) -> dict:
    """Parse perf stat stderr output into a dict."""
    stats = {}

    for line in output.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or "Performance counter" in line:
            continue

        # Pattern: "1,234,567      instructions" or "1,234,567      instructions:u"
        match = re.match(r"([\d,\.]+)\s+(\S+)", line)
        if match:
            value_str = match.group(1).replace(",", "")
            name = match.group(2).rstrip(":u").rstrip(":k")
            try:
                value = float(value_str)
                stats[name] = value
            except ValueError:
                pass

        # Pattern for ratios: "# 0.95 insn per cycle"
        if "insn per cycle" in line:
            match = re.search(r"#\s+([\d\.]+)\s+insn per cycle", line)
            if match:
                stats["ipc"] = float(match.group(1))

    return stats


def print_analysis(stats: dict):
    """Print analysis and optimization recommendations."""

    print(f"\n{'=' * 70}")
    print(f"Hardware Counter Analysis")
    print(f"{'=' * 70}\n")

    # IPC
    ipc = stats.get("ipc", 0)
    if ipc == 0 and stats.get("instructions", 0) > 0 and stats.get("cycles", 0) > 0:
        ipc = stats["instructions"] / stats["cycles"]

    print(f"  IPC:                 {ipc:.2f}")

    # Cache stats
    l1_loads = stats.get("L1-dcache-loads", 0)
    l1_misses = stats.get("L1-dcache-load-misses", 0)
    l1_miss_rate = (l1_misses / l1_loads * 100) if l1_loads > 0 else 0

    llc_loads = stats.get("LLC-loads", 0)
    llc_misses = stats.get("LLC-load-misses", 0)
    llc_miss_rate = (llc_misses / llc_loads * 100) if llc_loads > 0 else 0

    print(f"  L1 miss rate:        {l1_miss_rate:.1f}%")
    print(f"  LLC miss rate:       {llc_miss_rate:.1f}%")

    # Branch stats
    branches = stats.get("branch-instructions", 0)
    branch_misses = stats.get("branch-misses", 0)
    branch_miss_rate = (branch_misses / branches * 100) if branches > 0 else 0
    print(f"  Branch miss rate:    {branch_miss_rate:.1f}%")

    print()

    # Recommendations
    print(f"{'=' * 70}")
    print(f"Optimization Recommendations")
    print(f"{'=' * 70}\n")

    recommendations = []

    if ipc < 1.0:
        recommendations.append(
            ">> IPC < 1.0: Memory-bound or dependency-bound.\n"
            "   - Add prefetch instructions (_mm_prefetch with _MM_HINT_T0 or _MM_HINT_T1)\n"
            "   - Reduce cache blocking tile size\n"
            "   - Check for false sharing in OpenMP parallel regions\n"
            "   Reference: references/memory_patterns.yaml"
        )
    elif ipc < 2.0:
        recommendations.append(
            ">> IPC 1.0-2.0: Moderate compute efficiency.\n"
            "   - Try loop unrolling (#pragma GCC unroll 4)\n"
            "   - Ensure FMA instructions are being generated (_mm512_fmadd_ps)\n"
            "   Reference: references/simd_optimization_patterns.yaml"
        )
    else:
        recommendations.append(
            ">> IPC >= 2.0: Good compute efficiency.\n"
            "   - Focus on algorithmic improvements rather than micro-optimization"
        )

    if l1_miss_rate > 10:
        recommendations.append(
            f">> L1 miss rate = {l1_miss_rate:.1f}%: High — cache blocking too large.\n"
            "   - Reduce tile size to fit working set in L1 (48KB per core)\n"
            "   - Add L1 prefetch: _mm_prefetch(ptr, _MM_HINT_T0)\n"
            "   Reference: references/memory_patterns.yaml"
        )

    if llc_miss_rate > 20:
        recommendations.append(
            f">> LLC miss rate = {llc_miss_rate:.1f}%: High — working set exceeds L3.\n"
            "   - Add cache blocking to keep working set within L2 budget (1MB, use 50%)\n"
            "   - Consider streaming stores for write-only data\n"
            "   Reference: references/memory_patterns.yaml"
        )

    if branch_miss_rate > 5:
        recommendations.append(
            f">> Branch miss rate = {branch_miss_rate:.1f}%: Consider branchless patterns.\n"
            "   - Use SIMD masking instead of scalar branches\n"
            "   - Use __builtin_expect for predictable branches"
        )

    if not recommendations:
        recommendations.append(
            ">> All counters look healthy. Focus on algorithmic improvements."
        )

    for rec in recommendations:
        print(f"  {rec}\n")


def main():
    parser = argparse.ArgumentParser(description="Profile CPU kernel with perf stat")
    parser.add_argument("--kernel-package", required=True, help="Kernel package name")
    parser.add_argument("--op", required=True, help="Kernel function path (e.g., my_kernel.forward)")
    parser.add_argument("--baseline", default=None, help="Baseline file for getting inputs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Profiled iterations")
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"CPU Kernel Profiler")
    print(f"{'=' * 70}")
    print(f"Kernel: {args.op}")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")

    if not _CFG.get("perf_stat_enabled", True):
        print("\n  perf_stat_enabled=false in config.yaml. Skipping.")
        sys.exit(0)

    if not _check_perf_available():
        print("\n  'perf' not found. Install linux-tools-common or run with perf_stat_enabled=false.")
        sys.exit(1)

    stats = run_perf_stat(args.kernel_package, args.op, args.warmup, args.iters, args.baseline)

    if stats:
        print_analysis(stats)
    else:
        print("\n  No stats collected. Check perf permissions (try: echo -1 > /proc/sys/kernel/perf_event_paranoid)")


if __name__ == "__main__":
    main()
