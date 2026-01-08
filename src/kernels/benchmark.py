import importlib.util
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.utils import disable_progress_bars

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

CENTRAL_BENCHMARKS_REPO = "huggingface/kernels-benchmarks"
BENCHMARK_PATHS = ["benchmarks/bench.py", "benchmark.py"]


def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate percentile from sorted data using linear interpolation."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    idx = (n - 1) * p / 100.0
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    weight = idx - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def _calculate_iqr_and_outliers(
    times_ms: list[float],
) -> tuple[float, float, float, int]:
    """Calculate Q1, Q3, IQR, and count outliers.

    Returns:
        Tuple of (q1, q3, iqr, outlier_count)
    """
    sorted_times = sorted(times_ms)
    q1 = _percentile(sorted_times, 25)
    q3 = _percentile(sorted_times, 75)
    iqr = q3 - q1

    # Outliers are values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = sum(1 for t in times_ms if t < lower_bound or t > upper_bound)

    return q1, q3, iqr, outliers


class Benchmark:
    """Base class for kernel benchmarks.

    Subclass this to create a benchmark script with automatic timing,
    verification, and reproducibility support.

    Example:
        class MyBenchmark(Benchmark):
            kernel_id = "kernels-community/activation"
            seed = 42

            def setup(self):
                self.x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
                self.out = torch.empty(128, 512, device="cuda", dtype=torch.float16)

            def benchmark_silu(self):
                self.kernel.silu_and_mul(self.out, self.x)

            def verify_silu(self):
                ref = torch.nn.functional.silu(self.x[..., :512]) * self.x[..., 512:]
                return torch.allclose(self.out, ref, atol=1e-3)
    """

    kernel_id: str | None = None  # Required: kernel repo ID to load
    seed: int | None = None  # Optional: seed for reproducibility

    def __init__(self):
        self.kernel = None

    def setup(self):
        """Override to set up tensors as instance attributes."""
        pass


@dataclass
class TimingResults:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    q1_ms: float = 0.0  # 25th percentile
    q3_ms: float = 0.0  # 75th percentile
    iqr_ms: float = 0.0  # Interquartile range (Q3 - Q1)
    outliers: int = 0  # Count of outliers (outside Q1-1.5*IQR to Q3+1.5*IQR)
    verified: Optional[bool] = (
        None  # None = no verify fn, True = passed, False = failed
    )


@dataclass
class MachineInfo:
    gpu: str
    cuda_version: str
    pytorch_version: str
    os: str
    cpu: str


@dataclass
class BenchmarkResult:
    timing_results: dict[str, TimingResults]  # workload name -> timing
    machine_info: MachineInfo
    kernel_commit_sha: str
    benchmark_script_path: str
    benchmark_script_sha: Optional[str] = None

    def to_payload(self) -> dict:
        # Build results array for multiple workloads
        results = []
        for name, timing in sorted(self.timing_results.items()):
            entry: dict = {
                "workload": name,
                "timingResults": {
                    "mean_ms": timing.mean_ms,
                    "std_ms": timing.std_ms,
                    "min_ms": timing.min_ms,
                    "max_ms": timing.max_ms,
                    "q1_ms": timing.q1_ms,
                    "q3_ms": timing.q3_ms,
                    "iqr_ms": timing.iqr_ms,
                    "outliers": timing.outliers,
                    "iterations": timing.iterations,
                },
            }
            if timing.verified is not None:
                entry["verified"] = timing.verified
            results.append(entry)

        payload = {
            "results": results,
            "machineInfo": {
                "gpu": self.machine_info.gpu,
                "cudaVersion": self.machine_info.cuda_version,
                "pytorchVersion": self.machine_info.pytorch_version,
                "os": self.machine_info.os,
                "cpu": self.machine_info.cpu,
            },
            "kernelCommitSha": self.kernel_commit_sha,
            "benchmarkScriptPath": self.benchmark_script_path,
        }
        if self.benchmark_script_sha:
            payload["benchmarkScriptSha"] = self.benchmark_script_sha
        return payload


def _print_results_table(results: dict[str, TimingResults]) -> None:
    """Print a formatted table of benchmark results to stderr."""
    # Parse workload names into (class, method) tuples
    parsed: list[tuple[str, str, str]] = []  # (full_name, class_name, method_name)
    for name in sorted(results.keys()):
        if "." in name:
            cls, method = name.rsplit(".", 1)
        else:
            cls, method = "", name
        parsed.append((name, cls, method))

    # Calculate column widths
    cls_w = max((len(p[1]) for p in parsed), default=9)
    cls_w = max(cls_w, 9)  # minimum "Benchmark" header
    method_w = max((len(p[2]) for p in parsed), default=8)
    method_w = max(method_w, 8)  # minimum "Workload" header
    num_w = 10
    out_w = 8

    # Build table border
    ref_w = 11  # "Ref Match" column
    col_widths = [cls_w, method_w, num_w, num_w, num_w, num_w, num_w, out_w, ref_w - 2]

    def border(left, sep, right):
        return left + sep.join("─" * (w + 2) for w in col_widths) + right

    print(file=sys.stderr)
    print(border("┌", "┬", "┐"), file=sys.stderr)
    print(
        f"│ {'Benchmark':<{cls_w}} │ {'Workload':<{method_w}} │ {'Mean(ms)':>{num_w}} │ {'Std(ms)':>{num_w}} │ "
        f"{'Min(ms)':>{num_w}} │ {'Max(ms)':>{num_w}} │ {'IQR(ms)':>{num_w}} │ {'Outliers':>{out_w}} │ Ref Match │",
        file=sys.stderr,
    )
    print(border("├", "┼", "┤"), file=sys.stderr)

    for full_name, cls, method in parsed:
        t = results[full_name]
        check = "✓" if t.verified else ("✗" if t.verified is False else "·")
        print(
            f"│ {cls:<{cls_w}} │ {method:<{method_w}} │ {t.mean_ms:>{num_w}.4f} │ {t.std_ms:>{num_w}.4f} │ "
            f"{t.min_ms:>{num_w}.4f} │ {t.max_ms:>{num_w}.4f} │ {t.iqr_ms:>{num_w}.4f} │ {t.outliers:>{out_w}} │     {check}     │",
            file=sys.stderr,
        )

    print(border("└", "┴", "┘"), file=sys.stderr)
    print(file=sys.stderr)


def _get_macos_chip() -> Optional[str]:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _get_macos_gpu() -> Optional[str]:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu_name = displays[0].get("sppci_model", "")
                if gpu_name:
                    return gpu_name
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _get_local_commit_sha(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
        )
        return result.stdout.strip() if result.returncode == 0 else "local"
    except FileNotFoundError:
        return "local"


def collect_machine_info() -> MachineInfo:
    gpu = "N/A"
    cuda_version = "N/A"
    pytorch_version = "N/A"
    system = platform.system()
    os_info = f"{system} {platform.release()}"
    cpu = platform.processor() or platform.machine() or "Unknown"

    if system == "Darwin":
        cpu = _get_macos_chip() or cpu
        gpu = _get_macos_gpu() or gpu

    if TORCH_AVAILABLE:
        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda or "N/A"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu = _get_macos_gpu() or "Apple MPS"
            cuda_version = "MPS"

    if system == "Linux" and gpu == "N/A":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu = result.stdout.strip().split("\n")[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return MachineInfo(
        gpu=gpu,
        cuda_version=cuda_version,
        pytorch_version=pytorch_version,
        os=os_info,
        cpu=cpu,
    )


def get_kernel_commit_sha(repo_id: str, revision: str) -> str:
    if re.match(r"^[0-9a-f]{40}$", revision):
        return revision
    sha = HfApi().repo_info(repo_id=repo_id, revision=revision).sha
    if sha is None:
        raise ValueError(f"Could not get commit SHA for {repo_id}@{revision}")
    return sha


def run_benchmark_class(
    benchmark_cls: type[Benchmark],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, TimingResults]:
    """Run all benchmark_* methods in a Benchmark subclass.

    Args:
        benchmark_cls: A subclass of Benchmark
        iterations: Number of timed iterations
        warmup: Number of warmup iterations

    Returns:
        Dict mapping workload names to TimingResults
    """
    results = {}

    # Find all benchmark_* methods
    benchmark_methods = [
        name
        for name in dir(benchmark_cls)
        if name.startswith("benchmark_") and callable(getattr(benchmark_cls, name))
    ]

    if not benchmark_methods:
        raise RuntimeError(f"No benchmark_* methods found in {benchmark_cls.__name__}")

    # Load kernel once for all workloads
    kernel = None
    if benchmark_cls.kernel_id:
        from kernels import get_kernel

        kernel = get_kernel(benchmark_cls.kernel_id)

    for method_name in benchmark_methods:
        workload_name = method_name.replace("benchmark_", "")

        # Create fresh instance for each workload
        instance = benchmark_cls()
        instance.kernel = kernel

        # Apply seed for reproducibility
        if instance.seed is not None:
            torch.manual_seed(instance.seed)
            random.seed(instance.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(instance.seed)

        # Run setup (workload-specific if available, else default)
        setup_name = f"setup_{workload_name}"
        setup_fn = getattr(instance, setup_name, None)
        if setup_fn is not None:
            setup_fn()
        else:
            instance.setup()

        benchmark_fn = getattr(instance, method_name)
        verify_name = f"verify_{workload_name}"
        verify_fn = getattr(instance, verify_name, None)

        # Run verification if present
        verified: Optional[bool] = None
        if verify_fn is not None:
            benchmark_fn()  # Populate output
            torch.cuda.synchronize()
            verified = verify_fn()
            if not verified:
                raise RuntimeError(f"Verification failed for {workload_name}")

        # Warmup
        for _ in range(warmup):
            benchmark_fn()
            torch.cuda.synchronize()

        # Timing
        times_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            benchmark_fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)

        mean_ms = sum(times_ms) / len(times_ms)
        variance = sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)
        std_ms = variance**0.5
        q1, q3, iqr, outlier_count = _calculate_iqr_and_outliers(times_ms)

        results[workload_name] = TimingResults(
            mean_ms=round(mean_ms, 4),
            std_ms=round(std_ms, 4),
            min_ms=round(min(times_ms), 4),
            max_ms=round(max(times_ms), 4),
            iterations=iterations,
            q1_ms=round(q1, 4),
            q3_ms=round(q3, 4),
            iqr_ms=round(iqr, 4),
            outliers=outlier_count,
            verified=verified,
        )

    return results


def discover_benchmark_classes(script_path: Path, cwd: Path) -> list[type[Benchmark]]:
    """Import a script and find all Benchmark subclasses.

    Args:
        script_path: Path to the benchmark script
        cwd: Working directory for script execution

    Returns:
        List of Benchmark subclasses found in the script
    """
    spec = importlib.util.spec_from_file_location("benchmark_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_module"] = module

    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)

    # Find all Benchmark subclasses
    classes = []
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, Benchmark)
            and obj is not Benchmark
        ):
            classes.append(obj)

    return classes


def discover_benchmark_script(
    repo_id: str,
    repo_path: Path,
    *,
    use_central: bool = False,
    custom_script: Optional[str] = None,
) -> tuple[Path, Path]:
    """
    Discover the benchmark script to run.

    Returns:
        Tuple of (script_path, working_directory)
    """
    if custom_script:
        script_path = repo_path / custom_script
        if not script_path.exists():
            print(
                f"Error: Benchmark script not found: {custom_script}", file=sys.stderr
            )
            sys.exit(1)
        return script_path, repo_path

    if use_central:
        kernel_name = repo_id.split("/")[-1]
        central_path = Path(
            snapshot_download(
                repo_id=CENTRAL_BENCHMARKS_REPO,
                allow_patterns=[f"benchmarks/{kernel_name}/*"],
            )
        )
        script_path = central_path / "benchmarks" / kernel_name / "bench.py"
        if not script_path.exists():
            print(f"Error: No central benchmark for '{kernel_name}'", file=sys.stderr)
            print(
                "Try running without --central to use kernel's own benchmarks",
                file=sys.stderr,
            )
            sys.exit(1)
        return script_path, central_path / "benchmarks" / kernel_name

    # Default: search in kernel repo
    for rel_path in BENCHMARK_PATHS:
        script_path = repo_path / rel_path
        if script_path.exists():
            return script_path, repo_path

    print(f"Error: No benchmark found in '{repo_id}'", file=sys.stderr)
    print("Tried: benchmarks/bench.py, benchmark.py", file=sys.stderr)
    print(
        "Add benchmarks/bench.py or use --central for well-known benchmarks",
        file=sys.stderr,
    )
    sys.exit(1)


def _run_benchmark_script_subprocess(
    script_path: Path, *, iterations: int, warmup: int, cwd: Path
) -> dict[str, TimingResults]:
    """Run a legacy benchmark script as a subprocess."""
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark script failed:\n{result.stderr}")

    try:
        # Find JSON in output - benchmark scripts may print debug info before JSON
        stdout = result.stdout.strip()
        json_str = None
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                json_str = line
                break

        if json_str is None:
            raise RuntimeError(f"No JSON found in benchmark output:\n{result.stdout}")

        output = json.loads(json_str)
        timing = output["timing_results"]
        return {
            "default": TimingResults(
                mean_ms=timing["mean_ms"],
                std_ms=timing["std_ms"],
                min_ms=timing["min_ms"],
                max_ms=timing["max_ms"],
                iterations=timing["iterations"],
            )
        }
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(
            f"Error parsing benchmark output: {e}\nStdout: {result.stdout}"
        )


def run_benchmark_script(
    script_path: Path, *, iterations: int, warmup: int, cwd: Path
) -> dict[str, TimingResults]:
    """Run a benchmark script and return timing results.

    Automatically detects whether the script uses the class-based format
    (subclassing Benchmark) or the legacy subprocess format.

    Returns:
        Dict mapping workload names to TimingResults.
        For legacy scripts, returns {"default": timing}.
    """
    print(f"Running {script_path.name}...", file=sys.stderr)

    # Check if script uses class-based format
    script_content = script_path.read_text()
    uses_class = "class " in script_content and "(Benchmark)" in script_content

    if uses_class:
        classes = discover_benchmark_classes(script_path, cwd)
        if not classes:
            raise RuntimeError("No Benchmark subclasses found in script")

        # Run all benchmarks from all classes
        all_results = {}
        for cls in classes:
            results = run_benchmark_class(cls, iterations=iterations, warmup=warmup)
            for name, timing in results.items():
                all_results[f"{cls.__name__}.{name}"] = timing

        return all_results
    else:
        # Legacy subprocess-based execution
        return _run_benchmark_script_subprocess(
            script_path, iterations=iterations, warmup=warmup, cwd=cwd
        )


def submit_benchmark(
    repo_id: str,
    result: BenchmarkResult,
    *,
    api_url: Optional[str] = None,
    token: Optional[str] = None,
) -> None:
    if token is None:
        token = get_token()
    if token is None:
        raise ValueError(
            "No HuggingFace token. Run `huggingface-cli login` or use --token"
        )

    endpoint = f"{api_url or 'https://huggingface.co'}/api/models/{repo_id}/benchmarks"
    response = requests.post(
        endpoint,
        json=result.to_payload(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    if not response.ok:
        print(f"Error {response.status_code}: {response.text}", file=sys.stderr)
    response.raise_for_status()


def run_benchmark(
    repo_id: str,
    *,
    script: Optional[str] = None,
    use_central: bool = False,
    revision: str = "main",
    local_dir: Optional[str] = None,
    iterations: int = 100,
    warmup: int = 10,
    api_url: Optional[str] = None,
    token: Optional[str] = None,
    upload: bool = False,
    output: Optional[str] = None,
    print_json: bool = False,
) -> BenchmarkResult:
    # Suppress progress bars for cleaner output (files are often cached)
    disable_progress_bars()

    if local_dir:
        repo_path = Path(local_dir)
        if not repo_path.exists():
            print(f"Error: Local directory not found: {local_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Using local directory: {repo_path}", file=sys.stderr)
        kernel_sha = _get_local_commit_sha(repo_path)
    else:
        print(f"Downloading {repo_id}@{revision}...", file=sys.stderr)
        repo_path = Path(snapshot_download(repo_id=repo_id, revision=revision))
        kernel_sha = get_kernel_commit_sha(repo_id, revision)

    script_full_path, cwd = discover_benchmark_script(
        repo_id, repo_path, use_central=use_central, custom_script=script
    )

    try:
        timing_results = run_benchmark_script(
            script_full_path, iterations=iterations, warmup=warmup, cwd=cwd
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Print results table
    _print_results_table(timing_results)

    # Store relative path for the result (or absolute if outside cwd)
    try:
        script_rel_path = str(script_full_path.relative_to(cwd))
    except ValueError:
        script_rel_path = str(script_full_path)

    result = BenchmarkResult(
        timing_results=timing_results,
        machine_info=collect_machine_info(),
        kernel_commit_sha=kernel_sha,
        benchmark_script_path=script_rel_path,
    )

    if output:
        with open(output, "w") as f:
            json.dump(result.to_payload(), f, indent=2)
        print(f"Results saved to: {output}", file=sys.stderr)

    if print_json:
        print(json.dumps(result.to_payload(), indent=2))

    if upload:
        submit_benchmark(repo_id=repo_id, result=result, api_url=api_url, token=token)
        print("Benchmark submitted successfully!", file=sys.stderr)
    else:
        print("Dry run - use --upload to submit results", file=sys.stderr)

    return result
