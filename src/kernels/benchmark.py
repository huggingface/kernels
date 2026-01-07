import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import HfApi, get_token, snapshot_download

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

CENTRAL_BENCHMARKS_REPO = "huggingface/kernels-benchmarks"
BENCHMARK_PATHS = ["benchmarks/bench.py", "benchmark.py"]


@dataclass
class TimingResults:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int


@dataclass
class MachineInfo:
    gpu: str
    cuda_version: str
    pytorch_version: str
    os: str
    cpu: str


@dataclass
class BenchmarkResult:
    timing_results: TimingResults
    machine_info: MachineInfo
    kernel_commit_sha: str
    benchmark_script_path: str
    benchmark_script_sha: Optional[str] = None

    def to_payload(self) -> dict:
        payload = {
            "timingResults": {
                "mean_ms": self.timing_results.mean_ms,
                "std_ms": self.timing_results.std_ms,
                "min_ms": self.timing_results.min_ms,
                "max_ms": self.timing_results.max_ms,
                "iterations": self.timing_results.iterations,
            },
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
    return HfApi().repo_info(repo_id=repo_id, revision=revision).sha


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
            print(f"Error: Benchmark script not found: {custom_script}", file=sys.stderr)
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
            print(
                f"Error: No central benchmark for '{kernel_name}'", file=sys.stderr
            )
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


def run_benchmark_script(
    script_path: Path, *, iterations: int, warmup: int, cwd: Path
) -> TimingResults:
    print(f"Running {script_path.name}...", file=sys.stderr)

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
        output = json.loads(result.stdout)
        timing = output["timing_results"]
        return TimingResults(
            mean_ms=timing["mean_ms"],
            std_ms=timing["std_ms"],
            min_ms=timing["min_ms"],
            max_ms=timing["max_ms"],
            iterations=timing["iterations"],
        )
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(
            f"Error parsing benchmark output: {e}\nStdout: {result.stdout}"
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
    dry_run: bool = False,
    output: Optional[str] = None,
) -> BenchmarkResult:
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

    # Store relative path for the result
    script_rel_path = str(script_full_path.relative_to(cwd))

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

    if dry_run:
        print("Dry run - skipping API submission", file=sys.stderr)
        print(json.dumps(result.to_payload(), indent=2))
    else:
        submit_benchmark(repo_id=repo_id, result=result, api_url=api_url, token=token)
        print("Benchmark submitted successfully!", file=sys.stderr)

    return result
