# kernels benchmark

Use `kernels benchmark` to run benchmark scripts shipped with a kernel
repository.

The command:

- Downloads the kernel repo at a specific **branch** or **version**
- Runs all `benchmarks/benchmark*.py` scripts
- Times each `benchmark_*` workload and prints a results table
- Optionally saves results as JSON Lines records

## Installation

`kernels benchmark` requires extra dependencies:

```bash
uv pip install 'kernels[benchmark]' # or pip install 'kernels[benchmark]'
```

## Example

```bash
kernels benchmark kernels-community/activation --version 1
```

Example output:

```text
Downloading kernels-community/activation@v1...
Running benchmark.py...

  GPU      NVIDIA GeForce RTX 4090
  CPU      x86_64
  OS       Linux 6.12.63
  PyTorch  2.14.0+cu130

  Running SiluWorkloads on cuda

┌───────────────┬────────────┬─────┬───────────┬──────────────┬───────────┬─────────┐
│ Benchmark     │ Workload   │   N │ Speedup   │   Median(ms) │   Ref(ms) │ Match   │
├───────────────┼────────────┼─────┼───────────┼──────────────┼───────────┼─────────┤
│ SiluWorkloads │ large      │ 100 │ 0.84x     │       1.7466 │    1.4702 │ ✓       │
│ SiluWorkloads │ medium     │ 100 │ 1.67x     │       0.226  │    0.3769 │ ✓       │
│ SiluWorkloads │ small      │ 100 │ 1.10x     │       0.032  │    0.035  │ ✓       │
└───────────────┴────────────┴─────┴───────────┴──────────────┴───────────┴─────────┘

Kernel: cedce2e  Benchmark: 5b53516
```

`Speedup` is the reference median divided by the kernel median. Pass
`--show-stats` to also show the mean, standard deviation, min, max, IQR and
outlier count of each workload, and a 95% confidence interval for the mean.

## Usage

You must specify which revision to benchmark, either via flags or with `@...` in
the repo id:

```bash
kernels benchmark <repo_id> --version <N>
kernels benchmark <repo_id> --branch <name>
kernels benchmark <repo_id>@v<N>
kernels benchmark <repo_id>@<branch>
```

## Examples

Benchmark a tagged kernel version:

```bash
kernels benchmark kernels-community/activation --version 1
```

Equivalent shorthand:

```bash
kernels benchmark kernels-community/activation@v1
```

Benchmark a branch:

```bash
kernels benchmark kernels-community/activation --branch main
```

Tune warmup and iteration count:

```bash
kernels benchmark kernels-community/activation@v1 --warmup 20 --iterations 200
```

Save results to a file:

```bash
kernels benchmark kernels-community/activation@v1 --output results.jsonl
```

Benchmark a local kernel checkout (must contain `benchmarks/`):

```bash
kernels benchmark ./my_kernel
```

## Output

- By default, a table is printed (timings in ms).
- `--output <file>.jsonl` writes one record per workload as JSON Lines.
- `--json` prints the same records to stdout.

### Result records

Each JSON Lines record is a flat row that describes one workload of one run:
what was benchmarked (`repo_id`, `revision`, `kernel_sha`, `benchmark_sha`,
`benchmark`, `workload`, `params`), where (`gpu`, `backend`, `cpu`, `os`,
`torch_version`, `kernels_version`, `driver_version`, `gpu_max_clock_mhz`,
`gpu_power_limit_w`), how (`iterations`, `warmup`, `timing_mode`), the GPU state
during the workload (`gpu_clock_mhz`, `gpu_mem_clock_mhz`, `gpu_power_draw_w`,
`gpu_temperature_c`, `gpu_throttled`, `hardware`), and the results (`mean_ms`,
`median_ms`, `std_ms`, `min_ms`, `max_ms`, `q1_ms`, `q3_ms`, `iqr_ms`,
`outliers`, `verified`, `ref_mean_ms`, `ref_median_ms`, `speedup`, and the raw
`samples_ms` and `ref_samples_ms`). Records from one run share a `run_id`.

`params` (set by the benchmark) and `hardware` (vendor-specific, with a `vendor`
key) are JSON objects whose keys differ between benchmarks and vendors. Every
other field is present in every record, `null` when unknown.

Because every record is self-contained, files from many runs can be concatenated
and loaded as a single table, for example with pandas:

```python
import glob

import pandas as pd

df = pd.concat(pd.read_json(f, lines=True) for f in glob.glob("results/*.jsonl"))
```

### GPU conditions

Clocks, power limits and throttling affect timings, but `kernels benchmark` does
not change them. They are recorded instead, so that they can be taken into
account when comparing runs. They are collected with `nvidia-smi` on NVIDIA GPUs
and `amd-smi` on AMD GPUs, and are `null` on other devices. To get more stable
results, lock them yourself before benchmarking, e.g.:

```bash
sudo nvidia-smi --lock-gpu-clocks=2520,2520
sudo nvidia-smi --power-limit=400
```

## Comparing runs

Save a run as JSON Lines and pass it as a baseline to a later run, for example
to check whether a kernel or builder change affected performance:

```bash
kernels benchmark ./my_kernel --output base.jsonl
# ... make changes and rebuild ...
kernels benchmark ./my_kernel --baseline base.jsonl
```

This compares the kernel against an earlier run of itself. `Speedup` in the
results table is different: it compares the kernel against its `verify_*`
reference in the same run.

Two saved runs can also be compared directly:

```bash
kernels benchmark-compare base.jsonl new.jsonl
```

Workloads are compared by their median time. A workload is reported as faster or
slower only when its median changed by more than `--threshold` (default `0.05`,
i.e. 5%) _and_ the interquartile ranges of the two runs do not overlap. A
warning is printed when the runs were made on a different GPU, backend, PyTorch
version, driver version, or power limit. A note is printed for workloads whose
`params` or timing mode differ, and for changed workloads whose sampled clock or
power draw differ.

Use `--fail-on-regression` to exit with a non-zero status when any workload got
slower, which is useful in CI.

## Writing Benchmark Scripts

Benchmark scripts must live under `benchmarks/` in the kernel repository and
match `benchmark*.py`. Each script should define one or more subclasses of
`kernels.benchmark.Benchmark`.

Minimal example (`benchmarks/benchmark_activation.py`):

```python
import torch

from kernels.benchmark import Benchmark


class ActivationBenchmark(Benchmark):
    seed = 0

    def setup(self):
        self.x = torch.randn(128, 1024, device=self.device, dtype=torch.float16)
        self.out = torch.empty(128, 512, device=self.device, dtype=torch.float16)
        self.params = {"shape": list(self.x.shape), "dtype": self.x.dtype}  # optional

    def benchmark_silu_and_mul(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_silu_and_mul(self):
        # Return reference tensor; runner compares with self.out
        return torch.nn.functional.silu(self.x[..., :512]) * self.x[..., 512:]
```

The runner will:

- Set `self.device` to the device of the active backend (`cuda` on CUDA and
  ROCm, `xpu`, `mps` on Metal, `npu` on CANN), so scripts do not pick it
  themselves
- Call `setup()` once per workload (or `setup_<workload>()` if present)
- Warm up (`--warmup`)
- Time `benchmark_<workload>()` for `--iterations`
- If `verify_<workload>()` exists, check that outputs match
  (`torch.allclose(..., atol=1e-2)`) and show a speedup vs the reference
  computation (reference median / kernel median), timed with the same loop as
  the kernel
- Store `self.params`, if set, as the workload's `params`

Set `cuda_graph = True` on a benchmark class to time CUDA graph replays of each
workload and its reference, which removes most of the launch overhead. This is
supported on CUDA and ROCm; other devices fall back to eager timing. The mode
used is recorded as `timing_mode`.

## Troubleshooting

- If the repo does not contain a `benchmarks/` directory (or no `benchmark*.py`
  files), the command exits with an error.
- If a benchmark script defines no `Benchmark` subclasses, the command exits
  with an error.
- If `verify_<workload>()` exists and the outputs do not match, the command
  exits with an error.
