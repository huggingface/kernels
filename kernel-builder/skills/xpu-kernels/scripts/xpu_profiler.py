#!/usr/bin/env python3
"""
Profile a Triton kernel using Intel VTune on XPU hardware.

Collects GPU hardware counters (OA metrics) and maps bottlenecks to
optimization patterns.  Use when speedup plateaus or you need guidance on
which optimization level to try next.

Usage:
    python scripts/xpu_profiler.py <triton_file> [--warmup 5] [--iters 20]

Examples:
    python scripts/xpu_profiler.py test_kernels/39_Gemm_Scale_BatchNorm_triton.py
    python scripts/xpu_profiler.py output/14_Gemm_Divide_Sum_Scaling_triton.py --iters 50
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from skills.config import load_config as _load_config

_CFG = _load_config()
VTUNE_BIN = _CFG["vtune_bin"]

# PyTorch runtime kernel patterns — these are NOT user compute kernels
_OVERHEAD_KERNEL_PATTERNS = [
    re.compile(r"VectorizedElementwiseKernel"),
    re.compile(r"UnrolledElementwiseKernel"),
    re.compile(r"zeCommandListAppendMemoryCopy"),
    re.compile(r"ReduceKernelEmptyFunctor"),
    re.compile(r"\[Outside any task\]"),
]

# Columns to request from the hotspots report (the key OA hardware counters)
# VTune OA counter groups conflict when L3 BW and LSC BW are requested
# together. We use two report passes to get all columns.
_HOTSPOTS_COLUMNS_PASS1 = ",".join(
    [
        "Computing Task:Total Time",
        "Computing Task:Average Time",
        "Computing Task:Instance Count",
        "Computing Task:SIMD Width",
        "Computing Task:Spill Memory Size",
        "Work Size:Global",
        "Work Size:Local",
        # XVE execution breakdown
        "XVE Array:Active",
        "XVE Array:Stalled",
        "XVE Array:Idle",
        # Occupancy: peak (auto-includes Work Size/SLM/Barriers limiters)
        # NOTE: cannot combine with "XVE Threads Occupancy" — VTune counter conflict
        "Peak XVE Threads Occupancy",
        # Memory bandwidth (GPU VRAM)
        "GPU Memory Bandwidth, GB/sec:Read",
        "GPU Memory Bandwidth, GB/sec:Write",
        # L3 cache (includes BW — conflicts with LSC BW)
        "GPU L3:Busy",
        "GPU L3:Stalled",
        "GPU L3:Miss Ratio",
        "GPU L3:Average Bandwidth, GB/s:Read",
        "GPU L3:Average Bandwidth, GB/s:Write",
        "GPU L3:Input Available",
        "GPU L3:Output Ready",
        # Load/Store cache: ratios + pipeline (no BW — conflicts with L3 BW)
        "GPU Load Store Cache:Miss Ratio",
        "GPU Load Store Cache:L3 Miss Ratio",
        "GPU Load Store Cache:Input Available",
        "GPU Load Store Cache:Output Ready",
        "GPU Load Store Cache:Partial Writes",
        # Instruction cache
        "GPU Instruction cache L3 Miss Ratio",
        # SLM and misc
        "GPU Shared Local Memory:Bank Conflicts",
        "TLB Misses",
    ]
)

# Second pass: LSC bandwidth + XVE Threads Occupancy (measured)
# These conflict with Pass 1 columns.
_HOTSPOTS_COLUMNS_PASS2 = ",".join(
    [
        "Computing Task:Total Time",
        "XVE Threads Occupancy",
        "GPU Load Store Cache:Average Bandwidth, GB/s:Read",
        "GPU Load Store Cache:Average Bandwidth, GB/s:Write",
    ]
)


def _is_overhead_kernel(name: str) -> bool:
    """Return True if *name* matches a known PyTorch runtime kernel."""
    for pat in _OVERHEAD_KERNEL_PATTERNS:
        if pat.search(name):
            return True
    return False


# ---------------------------------------------------------------------------
# Runner script generation
# ---------------------------------------------------------------------------


def generate_runner_script(
    triton_file: Path, warmup: int, iters: int, vtune_bin: str = "", result_dir: str = ""
) -> str:
    # When vtune_bin and result_dir are provided, use VTune CLI to
    # resume/pause collection so only the profiled loop is captured.
    if vtune_bin and result_dir:
        resume_pause = f"""
import subprocess
_VTUNE_BIN = "{vtune_bin}"
_RESULT_DIR = "{result_dir}"
def _vtune_cmd(cmd):
    subprocess.run([_VTUNE_BIN, "-command", cmd, "-r", _RESULT_DIR],
                   capture_output=True, timeout=30)
"""
        resume_call = "_vtune_cmd('resume')"
    else:
        resume_pause = ""
        resume_call = "pass  # no VTune pause/resume"

    return f"""\
import torch
import importlib.util
import sys
{resume_pause}
spec = importlib.util.spec_from_file_location("triton_kernel", "{triton_file.resolve()}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

device = torch.device("xpu")
init_inputs = mod.get_init_inputs()
model = mod.Model(*init_inputs).to(device).eval()

inputs = mod.get_inputs()
inputs = [inp.to(device) if hasattr(inp, 'to') else inp for inp in inputs]

# Warmup (collection paused)
with torch.no_grad():
    for _ in range({warmup}):
        _ = model(*inputs)
        torch.xpu.synchronize()

# Resume collection for profiled iterations
{resume_call}

with torch.no_grad():
    for _ in range({iters}):
        _ = model(*inputs)
        torch.xpu.synchronize()

"""


# ---------------------------------------------------------------------------
# VTune CSV parsing
# ---------------------------------------------------------------------------


def parse_vtune_summary_csv(csv_path: Path) -> tuple[dict[str, str], list[dict], list[dict]]:
    """Parse the VTune ``-R summary`` TSV report.

    Returns (scalar_metrics, gpu_tasks, host_tasks).
    """
    scalar_metrics: dict[str, str] = {}
    gpu_tasks: list[dict] = []
    host_tasks: list[dict] = []

    if not csv_path.exists():
        return scalar_metrics, gpu_tasks, host_tasks

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if not rows:
        return scalar_metrics, gpu_tasks, host_tasks

    TABLE_SECTIONS = {
        "Top Hotspots when GPU was idle",
        "Hottest Host Tasks",
        "Hottest GPU Computing Tasks",
    }
    INFO_SECTION = "Collection and Platform Info"
    RECO_SECTION = "Recommendations:"

    def _parse_table(start_idx: int) -> tuple[list[dict], int]:
        header_row = rows[start_idx]
        col_names = [c.strip() for c in header_row[1:]]
        result = []
        idx = start_idx + 1
        while idx < len(rows):
            row = rows[idx]
            if len(row) < 2:
                break
            name = row[1].strip()
            if (
                len(row) >= 3
                and not row[2].strip()
                and name in (TABLE_SECTIONS | {INFO_SECTION, RECO_SECTION})
            ):
                break
            if len(row) == 2 or (len(row) >= 3 and not row[2].strip()):
                if name and not any(c.strip() for c in row[2:]):
                    break
            vals = [c.strip() for c in row[1:]]
            entry = {}
            for j, col in enumerate(col_names):
                entry[col] = vals[j] if j < len(vals) else ""
            result.append(entry)
            idx += 1
        return result, idx

    idx = 1  # skip header
    while idx < len(rows):
        row = rows[idx]
        if len(row) < 2:
            idx += 1
            continue

        name = row[1].strip()
        has_value = len(row) >= 3 and row[2].strip()

        if name in TABLE_SECTIONS and not has_value:
            idx += 1
            if idx >= len(rows):
                break
            entries, idx = _parse_table(idx)
            if name == "Hottest GPU Computing Tasks":
                gpu_tasks = entries
            elif name == "Hottest Host Tasks":
                host_tasks = entries
            continue

        if name == INFO_SECTION and not has_value:
            idx += 1
            while idx < len(rows):
                r = rows[idx]
                if len(r) < 2:
                    idx += 1
                    continue
                rname = r[1].strip()
                if rname == RECO_SECTION or rname.startswith("Recommendations"):
                    break
                rval = r[2].strip() if len(r) >= 3 else ""
                if rval:
                    scalar_metrics[rname] = rval
                idx += 1
            continue

        if name == RECO_SECTION or name.startswith("Recommendations"):
            # Parse VTune recommendations section
            idx += 1
            while idx < len(rows):
                r = rows[idx]
                if len(r) < 2:
                    idx += 1
                    continue
                rname = r[1].strip()
                rval = r[2].strip() if len(r) >= 3 else ""
                if rval and rname:
                    scalar_metrics[f"_reco_{rname}"] = rval
                idx += 1
            continue

        if has_value:
            scalar_metrics[name] = row[2].strip()

        idx += 1

    return scalar_metrics, gpu_tasks, host_tasks


def parse_hotspots_csv(csv_path: Path) -> list[dict]:
    """Parse the VTune ``-R hotspots -group-by computing-task`` TSV report.

    Returns a list of per-kernel dicts with OA hardware counter columns.
    """
    if not csv_path.exists():
        return []

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    if len(rows) < 2:
        return []

    # VTune may prepend warning lines (e.g. "war:Column filter is ON.").
    # Find the actual header row — it starts with "Computing Task".
    header_idx = 0
    for i, row in enumerate(rows):
        if row and row[0].strip().startswith("Computing Task"):
            header_idx = i
            break

    headers = [h.strip() for h in rows[header_idx]]
    result = []
    for row in rows[header_idx + 1 :]:
        if not row or not row[0].strip():
            continue
        entry = {}
        for j, h in enumerate(headers):
            entry[h] = row[j].strip() if j < len(row) else ""
        result.append(entry)
    return result


def _extract(value: str) -> float | None:
    """Extract a numeric value, stripping %, units, commas."""
    value = value.strip().rstrip("%").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Primary kernel identification
# ---------------------------------------------------------------------------


def find_primary_kernel(gpu_tasks: list[dict]) -> dict | None:
    """Find the primary compute kernel, skipping PyTorch overhead kernels.

    Among non-overhead kernels with the same name, pick the variant with the
    highest total time (the autotune winner).  If ALL kernels are overhead,
    fall back to the one with the highest time.
    """
    candidates = []
    fallback = None
    fallback_time = 0.0

    for task in gpu_tasks:
        name = task.get("Computing Task", "")
        t = _extract(task.get("Computing Task:Total Time", task.get("Total Time", "")))
        if t is None or name.startswith("["):
            continue

        if t > fallback_time:
            fallback_time = t
            fallback = task

        if not _is_overhead_kernel(name):
            candidates.append((t, task))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return fallback


def aggregate_kernel_variants(gpu_tasks: list[dict]) -> list[dict]:
    """Group rows by kernel name and sum times / average metrics.

    VTune reports the same kernel name multiple times with different SIMD
    widths (autotune configurations).  This helper aggregates them for the
    summary display while keeping the per-variant details available.
    """
    by_name: dict[str, list[dict]] = {}
    for task in gpu_tasks:
        name = task.get("Computing Task", "")
        by_name.setdefault(name, []).append(task)

    result = []
    for name, variants in by_name.items():
        total_time = sum(
            _extract(v.get("Computing Task:Total Time", v.get("Total Time", ""))) or 0
            for v in variants
        )
        total_count = sum(
            int(_extract(v.get("Computing Task:Instance Count", v.get("Instance Count", ""))) or 0)
            for v in variants
        )
        # Use the variant with the highest time for representative metrics
        best = max(
            variants,
            key=lambda v: (
                _extract(v.get("Computing Task:Total Time", v.get("Total Time", ""))) or 0
            ),
        )
        agg = dict(best)  # copy best variant's metrics
        agg["_total_time"] = total_time
        agg["_total_count"] = total_count
        agg["_num_variants"] = len(variants)
        result.append(agg)
    result.sort(key=lambda x: x["_total_time"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------


def print_host_tasks(host_tasks: list[dict]):
    if not host_tasks:
        return
    print("\n  Hottest Host Tasks:\n")
    print(f"  {'Host Task':<45}  {'Time (s)':>10}  {'% Elapsed':>10}  {'Count':>6}")
    print(f"  {'-' * 76}")
    for task in host_tasks:
        name = task.get("Host Task", "")
        ttime = task.get("Task Time", "")
        pct = task.get("% of Elapsed Time(%)", "")
        count = task.get("Task Count", "")
        if len(name) > 45:
            name = name[:42] + "..."
        pct_str = f"{pct}%" if pct else ""
        print(f"  {name:<45}  {ttime:>10}  {pct_str:>10}  {count:>6}")


def _truncate(name: str, maxlen: int = 55) -> str:
    if name.startswith("["):
        return name
    return name if len(name) <= maxlen else name[: maxlen - 3] + "..."


def print_gpu_tasks_summary(gpu_tasks: list[dict], has_oa: bool):
    """Print a compact GPU computing tasks table."""
    if not gpu_tasks:
        return

    agg = aggregate_kernel_variants(gpu_tasks)
    N = 55

    print("\n  GPU Computing Tasks (by kernel name):\n")
    if has_oa:
        print(
            f"  {'Kernel':<{N}} {'Time':>7} {'Cnt':>5} {'Active':>7} {'Stall':>7} {'Idle':>7} {'Occ%':>6} {'MemR':>7} {'MemW':>7}"
        )
        print(f"  {'-' * (N + 62)}")
    else:
        print(f"  {'Kernel':<{N}} {'Time':>7} {'Cnt':>5} {'SIMD':>5} {'Occ%':>7} {'Util%':>7}")
        print(f"  {'-' * (N + 37)}")

    def _fmt_pct(val: str) -> str:
        v = _extract(val)
        return f"{v:.1f}%" if v is not None else ""

    def _fmt_f(val: str) -> str:
        v = _extract(val)
        return f"{v:.1f}" if v is not None else ""

    for a in agg:
        name = a.get("Computing Task", "")
        tag = " *" if _is_overhead_kernel(name) else ""
        dname = _truncate(name, N - len(tag)) + tag
        tt = f"{a['_total_time']:.4f}"
        cnt = str(a["_total_count"])

        if has_oa:
            active_s = _fmt_pct(a.get("XVE Array:Active(%)", ""))
            stall_s = _fmt_pct(a.get("XVE Array:Stalled(%)", ""))
            idle_s = _fmt_pct(a.get("XVE Array:Idle(%)", ""))
            occ_s = _fmt_pct(
                a.get("XVE Threads Occupancy(%)", "") or a.get("Peak XVE Threads Occupancy(%)", "")
            )
            memr_s = _fmt_f(a.get("GPU Memory Bandwidth, GB/sec:Read", ""))
            memw_s = _fmt_f(a.get("GPU Memory Bandwidth, GB/sec:Write", ""))
            print(
                f"  {dname:<{N}} {tt:>7} {cnt:>5} {active_s:>7} {stall_s:>7} {idle_s:>7} {occ_s:>6} {memr_s:>7} {memw_s:>7}"
            )
        else:
            simd = a.get("Computing Task:SIMD Width", a.get("SIMD Width", ""))
            occ = a.get("Peak XVE Threads Occupancy(%)", "")
            util = a.get("SIMD Utilization(%)", "")
            occ_s = f"{_extract(occ):.1f}%" if _extract(occ) is not None else occ
            util_s = f"{_extract(util):.1f}%" if _extract(util) is not None else util
            vn = a["_num_variants"]
            simd_s = simd if vn == 1 else f"{simd}({vn})"
            print(f"  {dname:<{N}} {tt:>7} {cnt:>5} {simd_s:>5} {occ_s:>7} {util_s:>7}")

    print("\n  (* = PyTorch runtime overhead kernel, not user compute)")


def print_primary_kernel_detail(primary: dict, has_oa: bool):
    """Print detailed metrics for the primary compute kernel."""
    name = primary.get("Computing Task", "")
    print(f"\n{'=' * 70}")
    print(f"Primary Kernel Analysis: {_truncate(name, 50)}")
    print(f"{'=' * 70}")

    def _row(label, key, unit="", fmt=".4f"):
        val = primary.get(key, "")
        v = _extract(val)
        if v is not None:
            print(f"  {label:<42} {v:{fmt}}{unit}")
        elif val:
            print(f"  {label:<42} {val}")

    _row("Total Time", "Computing Task:Total Time", "s")
    _row("Average Time", "Computing Task:Average Time", "s", ".6f")
    _row("Instance Count", "Computing Task:Instance Count", "", ".0f")
    _row("SIMD Width", "Computing Task:SIMD Width", "", ".0f")
    _row("Spill Memory Size", "Computing Task:Spill Memory Size", " bytes", ".0f")

    gs = primary.get("Work Size:Global", "")
    ls = primary.get("Work Size:Local", "")
    if gs or ls:
        print(f"  {'Work Size (Global / Local)':<42} {gs} / {ls}")

    if has_oa:
        print()
        _row("XVE Active", "XVE Array:Active(%)", "%", ".1f")
        _row("XVE Stalled", "XVE Array:Stalled(%)", "%", ".1f")
        _row("XVE Idle", "XVE Array:Idle(%)", "%", ".1f")
        _row("XVE Threads Occupancy (measured)", "XVE Threads Occupancy(%)", "%", ".1f")
        _row("Peak XVE Threads Occupancy", "Peak XVE Threads Occupancy(%)", "%", ".1f")
        # Occupancy limiters — show what's capping peak occupancy
        ws_lim = _extract(primary.get("Peak XVE Threads Occupancy:Work Size Limit(%)", ""))
        slm_lim = _extract(primary.get("Peak XVE Threads Occupancy:SLM Use Limit(%)", ""))
        bar_lim = _extract(primary.get("Peak XVE Threads Occupancy:Barriers Use Limit(%)", ""))
        if ws_lim is not None:
            limiter = min(
                ("Work Size (grid too small)", ws_lim),
                ("SLM Usage", slm_lim if slm_lim is not None else 100),
                ("Barriers", bar_lim if bar_lim is not None else 100),
                key=lambda x: x[1],
            )
            print(
                f"  {'Occupancy Limiters':<42} WorkSize={ws_lim:.0f}%  SLM={slm_lim:.0f}%  Barriers={bar_lim:.0f}%"
            )
            if limiter[1] < 100:
                print(f"  {'  -> Bottleneck':<42} {limiter[0]}")
        print()
        _row("GPU Memory BW Read", "GPU Memory Bandwidth, GB/sec:Read", " GB/s", ".1f")
        _row("GPU Memory BW Write", "GPU Memory Bandwidth, GB/sec:Write", " GB/s", ".1f")
        print()
        _row("L3 Busy", "GPU L3:Busy(%)", "%", ".1f")
        _row("L3 Stalled", "GPU L3:Stalled(%)", "%", ".1f")
        _row("L3 Cache Miss Ratio", "GPU L3:Miss Ratio(%)", "%", ".1f")
        _row("L3 BW Read", "GPU L3:Average Bandwidth, GB/s:Read", " GB/s", ".1f")
        _row("L3 BW Write", "GPU L3:Average Bandwidth, GB/s:Write", " GB/s", ".1f")
        _row("L3 Input Available", "GPU L3:Input Available(%)", "%", ".1f")
        _row("L3 Output Ready", "GPU L3:Output Ready(%)", "%", ".1f")
        print()
        _row("LSC Miss Ratio", "GPU Load Store Cache:Miss Ratio(%)", "%", ".1f")
        _row("LSC -> L3 Miss Ratio", "GPU Load Store Cache:L3 Miss Ratio(%)", "%", ".1f")
        _row("LSC BW Read", "GPU Load Store Cache:Average Bandwidth, GB/s:Read", " GB/s", ".1f")
        _row("LSC BW Write", "GPU Load Store Cache:Average Bandwidth, GB/s:Write", " GB/s", ".1f")
        _row("LSC Input Available", "GPU Load Store Cache:Input Available(%)", "%", ".1f")
        _row("LSC Output Ready", "GPU Load Store Cache:Output Ready(%)", "%", ".1f")
        _row("LSC Partial Writes", "GPU Load Store Cache:Partial Writes", "", ".0f")
        print()
        _row("Instruction Cache L3 Miss", "GPU Instruction cache L3 Miss Ratio(%)", "%", ".1f")
        _row("SLM Bank Conflicts", "GPU Shared Local Memory:Bank Conflicts", "", ".0f")
        _row("TLB Misses", "TLB Misses", "", ".0f")


def print_recommendations(
    primary: dict | None,
    gpu_tasks: list[dict],
    host_tasks: list[dict],
    scalar_metrics: dict[str, str],
    has_oa: bool,
):
    """Generate actionable recommendations grounded in KB optimization patterns.

    Every recommendation references a specific KB file and pattern ID.
    Thresholds are based on hardware counter semantics, not arbitrary cutoffs.
    """
    recommendations = []

    # --- Host overhead check ---
    # Grounded in: references/memory_patterns.yaml (no_device_to_host_scalar_sync)
    total_host_time = sum(_extract(t.get("Task Time", "")) or 0 for t in host_tasks)
    total_gpu_time = sum(
        _extract(t.get("Computing Task:Total Time", t.get("Total Time", ""))) or 0
        for t in gpu_tasks
        if not t.get("Computing Task", "").startswith("[")
    )

    if total_host_time > 0 and total_gpu_time > 0 and total_host_time > total_gpu_time * 2:
        recommendations.append(
            (
                f"Host overhead ({total_host_time:.3f}s) >> GPU compute ({total_gpu_time:.3f}s). "
                "CPU-side dominates: check for .item()/.cpu() in hot path, "
                "ensure weight packing runs at init time (not in forward()).",
                "references/memory_patterns.yaml (no_device_to_host_scalar_sync)",
            )
        )

    # --- Overhead kernel dominance ---
    # Grounded in: references/optimization_levels.yaml (level_2_bandwidth_reduction)
    # When PyTorch Fill/Copy/Cast ops dominate, it means data type conversion
    # is happening at runtime instead of at pack time.
    overhead_time = sum(
        _extract(t.get("Computing Task:Total Time", t.get("Total Time", ""))) or 0
        for t in gpu_tasks
        if _is_overhead_kernel(t.get("Computing Task", ""))
        and not t.get("Computing Task", "").startswith("[")
    )
    if total_gpu_time > 0 and overhead_time / total_gpu_time > 0.30:
        recommendations.append(
            (
                f"Overhead kernels consume {overhead_time / total_gpu_time * 100:.0f}% of GPU time "
                f"({overhead_time:.4f}s). These are PyTorch Fill/Copy/Cast ops. "
                "Pre-pack weights AND inputs to bf16 at init time to eliminate them.",
                "references/optimization_levels.yaml (level_2_bandwidth_reduction)",
            )
        )

    if primary is None:
        _print_reco_section(recommendations)
        return

    if has_oa:
        active = _extract(primary.get("XVE Array:Active(%)", ""))
        stalled = _extract(primary.get("XVE Array:Stalled(%)", ""))
        idle = _extract(primary.get("XVE Array:Idle(%)", ""))
        occupancy = _extract(primary.get("XVE Threads Occupancy(%)", "")) or _extract(
            primary.get("Peak XVE Threads Occupancy(%)", "")
        )
        l3_miss = _extract(primary.get("GPU L3:Miss Ratio(%)", ""))
        lsc_l3_miss = _extract(primary.get("GPU Load Store Cache:L3 Miss Ratio(%)", ""))
        spill = _extract(primary.get("Computing Task:Spill Memory Size", ""))

        # Occupancy limiters
        ws_lim = _extract(primary.get("Peak XVE Threads Occupancy:Work Size Limit(%)", ""))
        slm_lim = _extract(primary.get("Peak XVE Threads Occupancy:SLM Use Limit(%)", ""))
        bar_lim = _extract(primary.get("Peak XVE Threads Occupancy:Barriers Use Limit(%)", ""))

        # --- XVE Stall-dominated: memory/dependency bound ---
        # Grounded in: references/optimization_levels.yaml (level_2_bandwidth_reduction)
        # and references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)
        # When XVE spends more time stalled than active, the execution units
        # are waiting on memory. This maps to Level 2 bandwidth reduction.
        if stalled is not None and active is not None and stalled > active:
            recommendations.append(
                (
                    f"XVE Stalled ({stalled:.0f}%) > Active ({active:.0f}%): memory/dependency bound. "
                    "Use tensor descriptors for better address codegen, pre-pack to bf16 to halve bandwidth, "
                    "try tile swizzling for better L3 locality.",
                    "references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling) + "
                    "references/optimization_levels.yaml (level_2_bandwidth_reduction)",
                )
            )

        # --- XVE Idle-dominated: underutilization ---
        # Grounded in: references/persistent_kernel_patterns.yaml (persistent_kernel_basic_tile_loop)
        # High idle means the GPU has execution units with no work scheduled.
        # This happens when the grid is too small or there aren't enough warps.
        if idle is not None and active is not None and idle > 30 and active < 30:
            recommendations.append(
                (
                    f"XVE Idle ({idle:.0f}%) with Active only ({active:.0f}%): GPU underutilized. "
                    "Grid may be too small — increase tile count or use persistent kernel pattern.",
                    "references/persistent_kernel_patterns.yaml (persistent_kernel_basic_tile_loop)",
                )
            )

        # --- Occupancy limiters ---
        # Grounded in: references/xpu_optimizations.yaml (xpu_grf_mode, xpu_tile_swizzling)
        # Check BOTH measured occupancy AND peak occupancy limiters.
        # Peak < 100% means hardware CAN'T give full occupancy (structural limit).
        # Measured < peak means kernel isn't filling available slots (launch config).
        peak_occ = _extract(primary.get("Peak XVE Threads Occupancy(%)", ""))
        occ_limited = (occupancy is not None and occupancy < 50) or (
            peak_occ is not None and peak_occ < 100
        )

        if occ_limited:
            if (
                ws_lim is not None
                and ws_lim < 100
                and (slm_lim is None or ws_lim <= (slm_lim or 100))
            ):
                recommendations.append(
                    (
                        f"Peak Occupancy capped at {ws_lim:.0f}% by Work Size (grid too small). "
                        "Increase grid dimensions, use tile swizzling with GROUP_SIZE_M, "
                        "or try persistent kernel pattern to keep all XVEs busy.",
                        "references/xpu_optimizations.yaml (xpu_tile_swizzling) + "
                        "references/persistent_kernel_patterns.yaml",
                    )
                )
            elif slm_lim is not None and slm_lim < 100:
                recommendations.append(
                    (
                        f"Peak Occupancy capped at {slm_lim:.0f}% by SLM Usage. "
                        "Kernel uses too much shared local memory per work group. "
                        "Reduce tile sizes or try grf_mode='large' to trade SLM for registers.",
                        "references/xpu_optimizations.yaml (xpu_grf_mode)",
                    )
                )
            elif bar_lim is not None and bar_lim < 100:
                recommendations.append(
                    (
                        f"Peak Occupancy capped at {bar_lim:.0f}% by Barriers. "
                        "Too many barrier synchronizations. Reduce num_warps or restructure "
                        "the kernel to use fewer synchronization points.",
                        "references/xpu_optimizations.yaml (xpu_warp_count)",
                    )
                )
            elif occupancy is not None and occupancy < 50:
                recommendations.append(
                    (
                        f"XVE Occupancy {occupancy:.0f}%: low thread count on GPU. "
                        "Try larger tiles, more warps, or grf_mode='large' (256 registers).",
                        "references/xpu_optimizations.yaml (xpu_grf_mode)",
                    )
                )

        # --- High L3 miss: data streaming from VRAM ---
        # Grounded in: references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling)
        # Tensor descriptors produce better address codegen; tile swizzling improves L3 reuse
        # across neighboring work groups.
        if l3_miss is not None and l3_miss > 50:
            recommendations.append(
                (
                    f"L3 Miss Ratio {l3_miss:.0f}%: data is streaming from VRAM with poor reuse. "
                    "Use tensor descriptors (better address codegen) and tile swizzling "
                    "(improves L3 locality across neighboring work groups).",
                    "references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling)",
                )
            )

        # --- High LSC->L3 miss: L1 thrashing ---
        # Grounded in: references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)
        # and references/memory_patterns.yaml (mem_block_pointers)
        if lsc_l3_miss is not None and lsc_l3_miss > 50:
            recommendations.append(
                (
                    f"LSC->L3 Miss Ratio {lsc_l3_miss:.0f}%: L1 cache not capturing data. "
                    "Use tensor descriptors for structured access patterns. "
                    "Ensure coalesced access within work groups.",
                    "references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern) + "
                    "references/memory_patterns.yaml (mem_block_pointers)",
                )
            )

        # --- Register spill ---
        # Grounded in: references/memory_patterns.yaml (reduce_liveness_sink_load_and_prefetch)
        # and references/xpu_optimizations.yaml (xpu_grf_mode)
        # Spill means the kernel exceeds the register file. Two mitigations:
        # 1. Reduce live variable count (sink loads closer to use, prefetch early)
        # 2. Use large GRF mode (256 registers) to increase register budget
        if spill is not None and spill > 0:
            recommendations.append(
                (
                    f"Register Spill: {spill:.0f} bytes. Kernel exceeds register file capacity. "
                    "Reduce variable liveness: sink loads closer to tl.dot(), use tl.prefetch() "
                    "to warm cache without holding registers. Also try grf_mode='large' (256 regs).",
                    "references/memory_patterns.yaml (reduce_liveness_sink_load_and_prefetch) + "
                    "references/xpu_optimizations.yaml (xpu_grf_mode)",
                )
            )

        # --- Instruction cache misses ---
        # High instruction cache L3 miss ratio means the compiled kernel binary
        # is too large to stay in instruction cache. This typically happens with
        # heavily unrolled or very large tile kernels.
        icache_miss = _extract(primary.get("GPU Instruction cache L3 Miss Ratio(%)", ""))
        if icache_miss is not None and icache_miss > 30:
            recommendations.append(
                (
                    f"Instruction Cache L3 Miss {icache_miss:.0f}%: compiled kernel is too large. "
                    "Reduce tile sizes or unrolling to shrink kernel binary. "
                    "Smaller BLOCK_K or fewer autotune configs can help.",
                    "references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)",
                )
            )

    else:
        # Fallback: no OA data, use summary metrics only
        occupancy = _extract(primary.get("Peak XVE Threads Occupancy(%)", ""))

        if occupancy is not None and occupancy < 50:
            recommendations.append(
                (
                    f"Peak XVE Occupancy {occupancy:.0f}%: low. "
                    "Try larger tiles, more warps, or grf_mode='large'.",
                    "references/xpu_optimizations.yaml (xpu_grf_mode, xpu_tile_swizzling)",
                )
            )

    _print_reco_section(recommendations)


def _print_reco_section(recommendations):
    print(f"\n{'=' * 70}")
    print("Optimization Recommendations")
    print(f"{'=' * 70}")
    if recommendations:
        for msg, ref in recommendations:
            print(f"\n  >> {msg}")
            print(f"     Reference: {ref}")
    else:
        print("\n  No bottlenecks detected -- kernel looks well-optimized!")
        print("  If speedup is still below target, consider:")
        print("    - Level 3: Algebraic fusion (fold BN/scale into weights)")
        print("    - Level 4: Stream K / persistent kernels")


# ---------------------------------------------------------------------------
# VTune execution
# ---------------------------------------------------------------------------


def run_vtune_collection(
    vtune_bin: str,
    triton_file: Path,
    result_dir: str,
    summary_csv: Path,
    warmup: int,
    iters: int,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Run VTune gpu-offload collection and produce summary CSV."""
    runner = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="vtune_runner_", delete=False
    )
    runner.write(
        generate_runner_script(
            triton_file, warmup, iters, vtune_bin=vtune_bin, result_dir=result_dir
        )
    )
    runner.close()

    try:
        # Step 1: Collect with -start-paused (runner resumes/pauses around profiled loop)
        collect_cmd = [
            vtune_bin,
            "-collect",
            "gpu-offload",
            "-start-paused",
            "-r",
            result_dir,
            "--",
            sys.executable,
            runner.name,
        ]
        result = subprocess.run(collect_cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return result

        # Step 2: Generate summary report from collected data
        report_cmd = [
            vtune_bin,
            "-R",
            "summary",
            "-r",
            result_dir,
            "-format",
            "csv",
            "-csv-delimiter",
            "tab",
            "-report-output",
            str(summary_csv),
        ]
        subprocess.run(report_cmd, capture_output=True, text=True, timeout=120)
        return result
    finally:
        try:
            os.unlink(runner.name)
        except OSError:
            pass


def _run_single_hotspots_report(
    vtune_bin: str, result_dir: str, csv_path: Path, columns: str
) -> bool:
    """Run one VTune hotspots report pass. Returns True on success."""
    cmd = [
        vtune_bin,
        "-R",
        "hotspots",
        "-r",
        result_dir,
        "-group-by",
        "computing-task",
        "-format",
        "csv",
        "-csv-delimiter",
        "tab",
        "-column",
        columns,
        "-report-output",
        str(csv_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return False
    return csv_path.exists() and csv_path.stat().st_size > 100


def run_hotspots_report(vtune_bin: str, result_dir: str, hotspots_csv: Path) -> bool:
    """Run VTune hotspots reports to extract per-kernel OA hardware counters.

    Uses two passes because some OA counter groups conflict (e.g. L3 BW
    and LSC BW cannot be collected simultaneously).  Pass 2 results are
    merged into Pass 1 by kernel name.

    Returns True if at least Pass 1 produced useful data.
    """
    ok = _run_single_hotspots_report(vtune_bin, result_dir, hotspots_csv, _HOTSPOTS_COLUMNS_PASS1)
    if not ok:
        return False

    # Pass 2: supplementary columns (LSC BW, measured occupancy)
    pass2_csv = hotspots_csv.with_name(hotspots_csv.stem + "_pass2.csv")
    ok2 = _run_single_hotspots_report(vtune_bin, result_dir, pass2_csv, _HOTSPOTS_COLUMNS_PASS2)
    if ok2:
        _merge_pass2(hotspots_csv, pass2_csv)
        try:
            pass2_csv.unlink()
        except OSError:
            pass

    return True


def _merge_pass2(main_csv: Path, pass2_csv: Path):
    """Merge Pass 2 columns into the main hotspots CSV by kernel name."""
    pass2_tasks = parse_hotspots_csv(pass2_csv)
    if not pass2_tasks:
        return

    # Build lookup by kernel name
    p2_by_name: dict[str, dict] = {}
    for task in pass2_tasks:
        name = task.get("Computing Task", "")
        if name and name not in p2_by_name:
            p2_by_name[name] = task

    # Re-read main CSV, add pass2 columns, rewrite
    main_tasks = parse_hotspots_csv(main_csv)
    if not main_tasks:
        return

    # Find new columns from pass2 (skip "Computing Task" and duplicates)
    main_keys = set(main_tasks[0].keys()) if main_tasks else set()
    new_cols = [k for k in pass2_tasks[0].keys() if k not in main_keys and k != "Computing Task"]

    if not new_cols:
        return

    for task in main_tasks:
        name = task.get("Computing Task", "")
        p2 = p2_by_name.get(name, {})
        for col in new_cols:
            task[col] = p2.get(col, "")

    # Rewrite main CSV
    all_cols = list(main_tasks[0].keys())
    with open(main_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(all_cols)
        for task in main_tasks:
            writer.writerow([task.get(c, "") for c in all_cols])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile Triton kernel with Intel VTune on XPU")
    parser.add_argument("triton_file", type=Path, help="Triton kernel implementation")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Profiled iterations")
    parser.add_argument("--timeout", type=int, default=300, help="VTune timeout (s)")
    args = parser.parse_args()

    if not args.triton_file.exists():
        print(f"Error: Triton file not found: {args.triton_file}")
        sys.exit(1)

    vtune_bin = os.environ.get("VTUNE_BIN", VTUNE_BIN)
    if not shutil.which(vtune_bin) and not Path(vtune_bin).is_file():
        print(f"Error: VTune binary not found at: {vtune_bin}")
        sys.exit(1)

    kernel_name = args.triton_file.stem
    timestamp = int(time.time())
    result_dir = f"/tmp/vtune_result_{kernel_name}_{timestamp}"
    summary_csv = Path(f"/tmp/vtune_{kernel_name}_{timestamp}_summary.csv")
    hotspots_csv = Path(f"/tmp/vtune_{kernel_name}_{timestamp}_hotspots.csv")

    print(f"\n{'=' * 70}")
    print("VTune Profiling Configuration")
    print(f"{'=' * 70}")
    print(f"  Triton kernel:  {args.triton_file}")
    print("  Collection:     gpu-offload (with OA hardware counters)")
    print(f"  Warmup iters:   {args.warmup}")
    print(f"  Profiled iters: {args.iters}")
    print(f"  Timeout:        {args.timeout}s")

    # --- Step 1: Run VTune collection ---
    print(f"\n{'=' * 70}")
    print("Running VTune Collection...")
    print(f"{'=' * 70}")

    try:
        result = run_vtune_collection(
            vtune_bin,
            args.triton_file,
            result_dir,
            summary_csv,
            args.warmup,
            args.iters,
            args.timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"\nError: VTune collection timed out after {args.timeout}s")
        sys.exit(1)

    if result.returncode != 0:
        print(f"\nError: VTune exited with code {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[:2000]}")
        sys.exit(1)

    # --- Step 2: Parse summary report ---
    scalar_metrics, gpu_tasks, host_tasks = parse_vtune_summary_csv(summary_csv)

    # --- Step 3: Run hotspots report for OA hardware counters ---
    has_oa = False
    hotspot_tasks: list[dict] = []
    if os.path.isdir(result_dir):
        ok = run_hotspots_report(vtune_bin, result_dir, hotspots_csv)
        if ok:
            hotspot_tasks = parse_hotspots_csv(hotspots_csv)
            # Check if OA columns are actually populated
            if hotspot_tasks:
                sample = hotspot_tasks[0]
                has_oa = bool(_extract(sample.get("XVE Array:Active(%)", "")))

    # Use hotspot_tasks if available (richer data), else fall back to summary gpu_tasks
    display_tasks = hotspot_tasks if hotspot_tasks else gpu_tasks

    # --- Step 4: Display results ---
    print(f"\n{'=' * 70}")
    print("VTune Profiling Results")
    print(f"{'=' * 70}")

    # Platform info (compact)
    gpu_name = scalar_metrics.get("Name", "")
    xve_count = scalar_metrics.get("XVE Count", "")
    max_freq = scalar_metrics.get("Max Core Frequency", "")
    if gpu_name:
        freq_ghz = f"{int(max_freq) / 1e9:.1f} GHz" if _extract(max_freq) else ""
        print(f"\n  GPU: {gpu_name}  XVEs: {xve_count}  Max Freq: {freq_ghz}")

    elapsed = scalar_metrics.get("Elapsed Time", "")
    gpu_pct = scalar_metrics.get("GPU Time, % of Elapsed time", "")
    if elapsed:
        print(f"  Elapsed: {elapsed}s  GPU Time: {gpu_pct}% of elapsed")

    # VTune's own recommendations (XVE Stalled/Idle)
    xve_stall_reco = scalar_metrics.get("_reco_XVE Array Stalled/Idle", "")
    if xve_stall_reco:
        # Extract the percentage
        match = re.search(r"([\d.]+)", xve_stall_reco)
        if match:
            print(f"  XVE Array Stalled/Idle: {match.group(1)}% of GPU busy time")

    print_host_tasks(host_tasks)
    print_gpu_tasks_summary(display_tasks, has_oa)

    # --- Step 5: Primary kernel detail ---
    primary = find_primary_kernel(display_tasks)
    if primary:
        print_primary_kernel_detail(primary, has_oa)

    # --- Step 6: Recommendations ---
    print_recommendations(primary, display_tasks, host_tasks, scalar_metrics, has_oa)

    print(f"\n{'=' * 70}")
    print("Profiling Complete")
    print(f"{'=' * 70}")
    print(f"  Summary CSV:    {summary_csv}")
    if hotspot_tasks:
        print(f"  Hotspots CSV:   {hotspots_csv}")
    if os.path.isdir(result_dir):
        print(f"  VTune result:   {result_dir}")
    if not has_oa:
        print("\n  Note: OA hardware counters not available.")
        print("  To enable: echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid")
    print()


if __name__ == "__main__":
    main()
