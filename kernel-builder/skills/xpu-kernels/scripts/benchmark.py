#!/usr/bin/env python3
"""
Benchmark Triton kernel against a baseline (PyTorch or Triton).

Validates correctness and measures performance using ai-bench.

Usage:
    python scripts/benchmark.py <baseline_file> <triton_file> [--spec <spec.yaml>] [--device <xpu|cuda>] [--ci] [--triton-baseline] [--baseline-us 123.45]

Examples:
    python scripts/benchmark.py test_kernels/14_Gemm_Divide_Sum_Scaling_pytorch.py output/14_Gemm_Divide_Sum_Scaling_triton.py
    python scripts/benchmark.py test_kernels/14_Gemm_Divide_Sum_Scaling_triton.py output/14_optimized_triton.py --triton-baseline
    python scripts/benchmark.py baseline.py triton.py --baseline-us 123.45  # skip baseline perf, use cached value
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_module(filepath: Path, module_name: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _determine_spec_type(spec_file: Path, ci: bool):
    """Determine which spec key to use for ai-bench."""
    from ai_bench.harness import core as ai_hc

    if ci:
        return ai_hc.SpecKey.V_CI

    import yaml

    with open(spec_file) as f:
        raw_spec = yaml.safe_load(f)

    # Prefer bench-xpu, fall back to bench-gpu, then ci
    if "bench-xpu" in raw_spec:
        spec_key = "bench-xpu"
    elif str(ai_hc.SpecKey.V_BENCH_GPU) in raw_spec:
        spec_key = str(ai_hc.SpecKey.V_BENCH_GPU)
    else:
        print("  No benchmark spec found (bench-xpu or bench-gpu). Falling back to CI mode.")
        spec_key = str(ai_hc.SpecKey.V_CI)

    if spec_key == str(ai_hc.SpecKey.V_CI):
        return ai_hc.SpecKey.V_CI
    elif spec_key == str(ai_hc.SpecKey.V_BENCH_GPU):
        return ai_hc.SpecKey.V_BENCH_GPU
    else:
        # bench-xpu — pass as string
        return spec_key


# ---------------------------------------------------------------------------
# Correctness check via ai-bench
# ---------------------------------------------------------------------------
def run_correctness(
    pytorch_file: Path, triton_file: Path, spec_file: Path | None, device_str: str
) -> bool:
    """Validate numerical equivalence using ai-bench."""
    if spec_file and spec_file.exists():
        return _run_correctness_with_spec(pytorch_file, triton_file, spec_file, device_str)
    else:
        return _run_correctness_no_spec(pytorch_file, triton_file, device_str)


def _run_correctness_with_spec(
    pytorch_file: Path, triton_file: Path, spec_file: Path, device_str: str
) -> bool:
    """Correctness check using ai-bench spec infrastructure."""
    try:
        import torch
        from ai_bench.harness import core as ai_hc
        from ai_bench.harness.runner.benchmark_compare import (
            check_correctness,
            copy_model_weights,
            set_all_seeds,
        )
        from ai_bench.harness.runner.kernel_runner import KernelRunner
    except ImportError as e:
        print(f"  Could not import ai_bench: {e}")
        print(f"  Install ai-bench: pip install -r scripts/requirements.txt")
        return False

    device = torch.device(device_str)

    # Use the same spec type as performance (bench-xpu / bench-gpu), fall back to CI
    spec_type = _determine_spec_type(spec_file, ci=False)
    print(f"  Using spec type: {spec_type}")

    runner = KernelRunner(
        spec_type=spec_type,
        device=device,
        backend=ai_hc.Backend.PYTORCH,
    )

    # Load spec
    spec = runner.load_spec(spec_file)

    if str(spec_type) not in spec:
        print(
            f"  Spec type '{spec_type}' not in {spec_file.name}, falling back to no-spec correctness check"
        )
        return _run_correctness_no_spec(pytorch_file, triton_file, device_str)

    variants = runner.get_spec_variants(spec)

    spec_inputs = runner.get_spec_inputs(spec)
    spec_inits = runner.get_spec_inits(spec)

    # Load model classes
    pytorch_model_cls = runner.load_model(pytorch_file)
    triton_model_cls = runner.load_model(triton_file)

    if pytorch_model_cls is None:
        print(f"  Could not load PyTorch model from {pytorch_file}")
        return False
    if triton_model_cls is None:
        print(f"  Could not load Triton model from {triton_file}")
        return False

    print(f"  Found {len(variants)} variant(s)")

    all_correct = True
    for i, variant in enumerate(variants):
        set_all_seeds(123)

        # Log variant details
        dims = variant.get("dims", {})
        dtype = variant.get("dtype", "unknown")
        print(f"  Variant {i}: dtype={dtype}, dims={dims}")

        rtol = ai_hc.get_rtol(variant)
        atol = ai_hc.get_atol(variant)
        has_explicit_tol = "atol" in variant or "rtol" in variant
        # Only apply bf16 atol floor when no explicit spec tolerance was set
        if not has_explicit_tol and atol < 1e-2:
            atol = 1e-2

        # Instantiate models with variant-specific init params (eval mode for deterministic BN)
        pytorch_model = runner.init_model(pytorch_model_cls, variant, spec_inits).eval()
        triton_model = runner.init_model(triton_model_cls, variant, spec_inits).eval()

        # Sync weights from reference to optimized
        copy_model_weights(pytorch_model, triton_model)

        # Create inputs from spec
        args = ai_hc.get_inputs(variant, spec_inputs, device=device)

        with torch.no_grad():
            pytorch_output = pytorch_model(*args)
            triton_output = triton_model(*args)

        # Cast to common dtype (fp32) for comparison — models may use different output dtypes
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]
        if isinstance(triton_output, tuple):
            triton_output = triton_output[0]
        pytorch_output = pytorch_output.float()
        triton_output = triton_output.float()

        correct = check_correctness(pytorch_output, triton_output, rtol, atol)
        status = "PASS" if correct else "FAIL"
        print(f"  Variant {i}: {status} (rtol={rtol:.1e}, atol={atol:.1e})")

        if not correct:
            all_correct = False

    return all_correct


def _run_correctness_no_spec(pytorch_file: Path, triton_file: Path, device_str: str) -> bool:
    """Fallback correctness check without spec file."""
    try:
        import torch
        from ai_bench.harness.runner.benchmark_compare import (
            check_correctness,
            copy_model_weights,
            set_all_seeds,
        )
    except ImportError as e:
        print(f"  Could not import ai_bench: {e}")
        print(f"  Install ai-bench: pip install -r scripts/requirements.txt")
        return False

    device = torch.device(device_str)

    # Default tolerances (bf16 accumulation typically needs atol >= 1e-2)
    rtol, atol = 1e-2, 1e-2

    # Load modules directly
    pytorch_mod = _load_module(pytorch_file, "pytorch_ref")
    triton_mod = _load_module(triton_file, "triton_kernel")

    set_all_seeds(123)

    init_inputs = pytorch_mod.get_init_inputs()
    pytorch_model = pytorch_mod.Model(*init_inputs).to(device).eval()
    triton_model = triton_mod.Model(*init_inputs).to(device).eval()

    # Sync weights
    copy_model_weights(pytorch_model, triton_model)

    # Run both models
    inputs = pytorch_mod.get_inputs()
    inputs = [inp.to(device) if hasattr(inp, "to") else inp for inp in inputs]

    with torch.no_grad():
        pytorch_output = pytorch_model(*inputs)
        triton_output = triton_model(*inputs)

    # Cast to common dtype (fp32) for comparison
    if isinstance(pytorch_output, tuple):
        pytorch_output = pytorch_output[0]
    if isinstance(triton_output, tuple):
        triton_output = triton_output[0]
    pytorch_output = pytorch_output.float()
    triton_output = triton_output.float()

    correct = check_correctness(pytorch_output, triton_output, rtol, atol)
    print(f"  Result: {'PASS' if correct else 'FAIL'} (rtol={rtol:.1e}, atol={atol:.1e})")

    return correct


# ---------------------------------------------------------------------------
# Performance benchmark via ai-bench
# ---------------------------------------------------------------------------
def find_spec_file(triton_file: Path) -> Path | None:
    """Derive the YAML spec path from the same directory as the input file."""
    stem = triton_file.stem
    for suffix in ("_triton", "_optimized", "_opt", "_pytorch"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parent = triton_file.parent
    for candidate_stem in (stem, stem + "_pytorch"):
        candidate = parent / (candidate_stem + ".yaml")
        if candidate.exists():
            return candidate
    return None


def run_performance(
    pytorch_file: Path,
    triton_file: Path,
    spec_file: Path,
    device_str: str,
    ci: bool,
    triton_baseline: bool = False,
    baseline_us: list[float] | None = None,
) -> bool:
    """Benchmark baseline vs optimized Triton using ai-bench KernelRunner.

    If baseline_us is provided (list of floats, one per variant), the baseline
    performance measurement is skipped and the cached values are used instead.
    """
    try:
        import ai_bench
        import torch
        from ai_bench.harness import core as ai_hc
        from ai_bench.harness.runner.kernel_runner import KernelRunner
    except ImportError as e:
        print(f"  Could not import ai_bench: {e}")
        print(f"  Install ai-bench: pip install -r scripts/requirements.txt")
        return False

    device = torch.device(device_str)
    spec_type = _determine_spec_type(spec_file, ci)

    baseline_backend = ai_hc.Backend.TRITON if triton_baseline else ai_hc.Backend.PYTORCH
    baseline_label = "Triton baseline" if triton_baseline else "PyTorch baseline"
    cached = baseline_us is not None

    # --- Run or cache baseline ---
    if cached:
        print(f"\n  Using cached {baseline_label} times: {baseline_us}")
        pytorch_stats = [SimpleNamespace(meas_us=val) for val in baseline_us]
    else:
        print(f"\n  Running {baseline_label}...")
        pytorch_runner = KernelRunner(
            spec_type=spec_type,
            device=device,
            backend=baseline_backend,
        )
        pytorch_stats = pytorch_runner.run_kernel_spec(pytorch_file, spec_file)

        if pytorch_stats is None:
            print(f"  {baseline_label}: spec type '{spec_type}' not found in {spec_file.name}")
            return False

    # --- Run optimized Triton kernel ---
    print(f"  Running optimized Triton kernel...")
    triton_runner = KernelRunner(
        spec_type=spec_type,
        device=device,
        backend=ai_hc.Backend.TRITON,
    )
    triton_stats = triton_runner.run_kernel_spec(triton_file, spec_file)

    if triton_stats is None:
        print(f"  Triton kernel: spec type '{spec_type}' not found in {spec_file.name}")
        return False

    # Validate cached baseline count matches triton variant count
    if cached and len(pytorch_stats) != len(triton_stats):
        print(
            f"  Error: --baseline-us has {len(pytorch_stats)} value(s) but spec has "
            f"{len(triton_stats)} variant(s). Re-run without --baseline-us."
        )
        return False

    # --- CI mode: just validate both run without error ---
    if ci or spec_type == ai_hc.SpecKey.V_CI:
        print(
            f"  CI validation: PyTorch {'OK' if pytorch_stats is not None else 'FAILED'}, "
            f"Triton {'OK' if triton_stats is not None else 'FAILED'}"
        )
        return pytorch_stats is not None and triton_stats is not None

    # --- Report benchmark results ---
    if not pytorch_stats or not triton_stats:
        print("  No benchmark stats available.")
        return False

    if triton_baseline:
        baseline_col = "Triton BL (us)*" if cached else "Triton BL (us)"
    else:
        baseline_col = "PyTorch (us)*" if cached else "PyTorch (us)"
    print(f"\n  {'Variant':<8} {baseline_col:>16} {'Triton (us)':>14} {'Speedup':>10}")
    print(f"  {'-' * 52}")

    all_faster = True
    for i, (pt_stat, tr_stat) in enumerate(zip(pytorch_stats, triton_stats)):
        speedup = pt_stat.meas_us / tr_stat.meas_us if tr_stat.meas_us > 0 else 0
        marker = "+" if speedup >= 1.0 else "-"
        if speedup < 1.0:
            all_faster = False
        print(
            f"  {i:<8} {pt_stat.meas_us:>16.2f} {tr_stat.meas_us:>14.2f} {speedup:>9.2f}x {marker}"
        )

    if cached:
        print(f"\n  * baseline cached from prior trial")

    return all_faster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton kernel against a baseline (PyTorch or Triton)"
    )
    parser.add_argument(
        "pytorch_file",
        type=Path,
        help="Baseline file (PyTorch by default, or Triton with --triton-baseline)",
    )
    parser.add_argument("triton_file", type=Path, help="Optimized Triton kernel implementation")
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="YAML spec file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--device",
        default="xpu",
        help="Target device (default: xpu, always falls back to xpu)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: quick validation only, no benchmarking",
    )
    parser.add_argument(
        "--triton-baseline",
        action="store_true",
        help="Baseline file is a Triton kernel (use Backend.TRITON)",
    )
    parser.add_argument(
        "--baseline-us",
        type=str,
        default=None,
        help="Comma-separated cached baseline time(s) in microseconds (skips baseline perf run)",
    )
    args = parser.parse_args()

    # Parse --baseline-us into list of floats
    baseline_us = None
    if args.baseline_us is not None:
        try:
            baseline_us = [float(v.strip()) for v in args.baseline_us.split(",")]
        except ValueError:
            print(f"Error: --baseline-us must be comma-separated floats, got: {args.baseline_us}")
            sys.exit(1)

    if not args.pytorch_file.exists():
        print(f"Error: Baseline file not found: {args.pytorch_file}")
        sys.exit(1)
    if not args.triton_file.exists():
        print(f"Error: Triton file not found: {args.triton_file}")
        sys.exit(1)

    # Always fall back to XPU — this project only targets Intel XPU
    if args.device != "xpu":
        print(
            f"Warning: device '{args.device}' requested, falling back to 'xpu' (only XPU is supported)"
        )
        args.device = "xpu"

    spec_file = args.spec or find_spec_file(args.triton_file) or find_spec_file(args.pytorch_file)

    baseline_label = "Triton baseline" if args.triton_baseline else "PyTorch baseline"

    print(f"\n{'=' * 70}")
    print(f"Benchmark Configuration")
    print(f"{'=' * 70}")
    print(f"{baseline_label}:  {args.pytorch_file}")
    print(f"Triton kernel:    {args.triton_file}")
    print(f"Spec file:        {spec_file or '(none)'}")
    print(f"Device:           {args.device}")
    print(f"Mode:             {'CI' if args.ci else 'Benchmark'}")

    # --- Correctness ---
    print(f"\n{'=' * 70}")
    print(f"Correctness Check (ai-bench)")
    print(f"{'=' * 70}")
    correctness_passed = run_correctness(
        args.pytorch_file, args.triton_file, spec_file, args.device
    )
    print(f"\n  Result: {'PASSED' if correctness_passed else 'FAILED'}")

    # --- Performance ---
    performance_passed = None
    if spec_file and spec_file.exists():
        print(f"\n{'=' * 70}")
        print(f"Performance Benchmark (ai-bench)")
        print(f"{'=' * 70}")
        performance_passed = run_performance(
            args.pytorch_file,
            args.triton_file,
            spec_file,
            args.device,
            args.ci,
            triton_baseline=args.triton_baseline,
            baseline_us=baseline_us,
        )
        print(f"\n  Result: {'PASSED' if performance_passed else 'FAILED'}")
    else:
        print(f"\n  Skipping performance benchmark (no spec file found)")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"Summary")
    print(f"{'=' * 70}")
    print(f"Correctness: {'PASSED' if correctness_passed else 'FAILED'}")
    if performance_passed is not None:
        print(f"Performance: {'PASSED' if performance_passed else 'FAILED'}")
    else:
        print(f"Performance: SKIPPED (no spec file)")
    print()

    if correctness_passed and (performance_passed is not False):
        print("All checks passed!")
        sys.exit(0)
    else:
        print("Some checks FAILED - see output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
