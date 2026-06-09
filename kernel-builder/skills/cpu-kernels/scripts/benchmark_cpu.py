#!/usr/bin/env python3
"""
Benchmark CPU kernel against PyTorch baseline.

Validates correctness and measures performance using torch.utils.benchmark.

Usage:
    python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.forward
    python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.forward --baseline-us 123.45

Examples:
    # First trial — measures both baseline and kernel:
    python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.apply_rms_norm

    # Subsequent trials — use cached baseline:
    python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.apply_rms_norm --baseline-us 45.2
"""

import argparse
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

import torch


def _load_module(filepath: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_kernel_func(kernel_package: str, op_path: str):
    """Load a function from a kernel package.

    op_path is like 'my_kernel.apply_rms_norm' or 'my_kernel.forward'.
    """
    parts = op_path.split(".")
    if len(parts) < 2:
        print(f"Error: --op should be package.function (e.g., my_kernel.forward)")
        sys.exit(1)

    mod = importlib.import_module(parts[0])
    func = mod
    for attr in parts[1:]:
        func = getattr(func, attr)
    return func


def run_correctness(baseline_mod, kernel_func, device="cpu", atol=1e-2, rtol=1e-2):
    """Validate numerical equivalence between baseline and kernel."""
    print(f"\n  Correctness Check (atol={atol}, rtol={rtol})")

    try:
        # Get inputs from baseline
        if hasattr(baseline_mod, "get_inputs"):
            inputs = baseline_mod.get_inputs()
        else:
            print("  Error: baseline must define get_inputs()")
            return False

        # Get reference output
        if hasattr(baseline_mod, "get_reference_output"):
            ref_output = baseline_mod.get_reference_output(*inputs)
        elif hasattr(baseline_mod, "Model"):
            init_inputs = baseline_mod.get_init_inputs() if hasattr(baseline_mod, "get_init_inputs") else []
            model = baseline_mod.Model(*init_inputs)
            model.eval()
            with torch.no_grad():
                ref_output = model(*inputs)
        else:
            print("  Error: baseline must define get_reference_output() or Model class")
            return False

        # Run kernel
        with torch.no_grad():
            kernel_output = kernel_func(*inputs)

        # Compare
        if isinstance(ref_output, tuple):
            ref_output = ref_output[0]
        if isinstance(kernel_output, tuple):
            kernel_output = kernel_output[0]

        ref_output = ref_output.float()
        kernel_output = kernel_output.float()

        if ref_output.shape != kernel_output.shape:
            print(f"  FAIL: Shape mismatch: ref={ref_output.shape}, kernel={kernel_output.shape}")
            return False

        max_diff = (ref_output - kernel_output).abs().max().item()
        mean_diff = (ref_output - kernel_output).abs().mean().item()

        correct = torch.allclose(ref_output, kernel_output, atol=atol, rtol=rtol)
        status = "PASS" if correct else "FAIL"
        print(f"  {status}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")

        if not correct:
            # Show where the differences are largest
            diff = (ref_output - kernel_output).abs()
            flat_idx = diff.argmax().item()
            idx = []
            remaining = flat_idx
            for dim in reversed(ref_output.shape):
                idx.insert(0, remaining % dim)
                remaining //= dim
            print(f"  Largest diff at index {tuple(idx)}: ref={ref_output.flatten()[flat_idx]:.6f}, kernel={kernel_output.flatten()[flat_idx]:.6f}")

        return correct

    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return False


def run_performance(baseline_mod, kernel_func, baseline_us=None, warmup=10, iters=100):
    """Benchmark baseline vs kernel using torch.utils.benchmark."""
    from torch.utils.benchmark import Timer

    print(f"\n  Performance Benchmark (warmup={warmup}, iters={iters})")

    # Get inputs
    if hasattr(baseline_mod, "get_inputs"):
        inputs = baseline_mod.get_inputs()
    else:
        print("  Error: baseline must define get_inputs()")
        return None, None, None

    # Baseline timing
    if baseline_us is not None:
        print(f"  Using cached baseline: {baseline_us:.2f} us")
        bl_us = baseline_us
    else:
        if hasattr(baseline_mod, "get_reference_output"):
            ref_func = baseline_mod.get_reference_output
        elif hasattr(baseline_mod, "Model"):
            init_inputs = baseline_mod.get_init_inputs() if hasattr(baseline_mod, "get_init_inputs") else []
            model = baseline_mod.Model(*init_inputs)
            model.eval()
            ref_func = lambda *args: model(*args)
        else:
            print("  Error: baseline must define get_reference_output() or Model class")
            return None, None, None

        bl_timer = Timer(
            stmt="ref_func(*inputs)",
            globals={"ref_func": ref_func, "inputs": inputs},
            label="Baseline",
            description="PyTorch",
            num_threads=torch.get_num_threads(),
        )
        bl_result = bl_timer.blocked_autorange(min_run_time=2.0)
        bl_us = bl_result.median * 1e6
        print(f"  Baseline: {bl_us:.2f} us (median)")

    # Kernel timing
    kr_timer = Timer(
        stmt="kernel_func(*inputs)",
        globals={"kernel_func": kernel_func, "inputs": inputs},
        label="Kernel",
        description="CPU Kernel",
        num_threads=torch.get_num_threads(),
    )
    kr_result = kr_timer.blocked_autorange(min_run_time=2.0)
    kr_us = kr_result.median * 1e6
    print(f"  Kernel:   {kr_us:.2f} us (median)")

    speedup = bl_us / kr_us if kr_us > 0 else 0
    marker = "+" if speedup >= 1.0 else "-"
    print(f"  Speedup:  {speedup:.2f}x {marker}")

    return bl_us, kr_us, speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU kernel against PyTorch baseline")
    parser.add_argument("baseline_file", type=Path, help="PyTorch baseline file")
    parser.add_argument("--kernel-package", required=True, help="Kernel package name (pip-installed)")
    parser.add_argument("--op", required=True, help="Kernel function path (e.g., my_kernel.forward)")
    parser.add_argument("--baseline-us", type=float, default=None, help="Cached baseline time in microseconds")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness")
    args = parser.parse_args()

    if not args.baseline_file.exists():
        print(f"Error: Baseline file not found: {args.baseline_file}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"CPU Kernel Benchmark")
    print(f"{'=' * 70}")
    print(f"Baseline:       {args.baseline_file}")
    print(f"Kernel package: {args.kernel_package}")
    print(f"Op:             {args.op}")
    print(f"Threads:        {torch.get_num_threads()}")

    # Load baseline
    baseline_mod = _load_module(args.baseline_file, "baseline")

    # Load kernel function
    try:
        kernel_func = _load_kernel_func(args.kernel_package, args.op)
    except Exception as e:
        print(f"\nError loading kernel: {e}")
        print(f"Make sure '{args.kernel_package}' is installed: pip install dist/*.whl --force-reinstall")
        sys.exit(1)

    # Correctness
    print(f"\n{'=' * 70}")
    print(f"Correctness")
    print(f"{'=' * 70}")
    correct = run_correctness(baseline_mod, kernel_func, atol=args.atol, rtol=args.rtol)
    print(f"\n  Result: {'PASSED' if correct else 'FAILED'}")

    # Performance
    print(f"\n{'=' * 70}")
    print(f"Performance")
    print(f"{'=' * 70}")
    bl_us, kr_us, speedup = run_performance(baseline_mod, kernel_func, baseline_us=args.baseline_us)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"Correctness: {'PASSED' if correct else 'FAILED'}")
    if speedup is not None:
        print(f"Baseline:    {bl_us:.2f} us")
        print(f"Kernel:      {kr_us:.2f} us")
        print(f"Speedup:     {speedup:.2f}x")
    print()

    if correct and speedup is not None and speedup >= 1.0:
        print("All checks passed!")
        sys.exit(0)
    else:
        print("Some checks FAILED - see output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
