# kernels benchmark

Use `kernels benchmark` to run benchmark scripts shipped with a kernel repository.

The command:

- Downloads the kernel repo at a specific **branch** or **version**
- Runs all `benchmarks/benchmark*.py` scripts
- Times each `benchmark_*` workload and prints a results table
- Optionally saves results as JSON

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

  GPU      Apple M3 Max (30 cores)
  CPU      Apple M3 Max
  OS       Darwin 25.2.0
  PyTorch  2.10.0

  Running SiluWorkloads on mps

┌───────────────┬────────────┬─────┬───────────┬────────────┬───────────┬───────────┬───────────┬───────────┬────────────┬───────────┬─────────┐
│ Benchmark     │ Workload   │   N │ Speedup   │   Mean(ms) │   Std(ms) │   Min(ms) │   Max(ms) │   IQR(ms) │   Outliers │   Ref(ms) │ Match   │
├───────────────┼────────────┼─────┼───────────┼────────────┼───────────┼───────────┼───────────┼───────────┼────────────┼───────────┼─────────┤
│ SiluWorkloads │ large      │ 100 │ 1.72x     │     6.5153 │    0.4343 │    6.2883 │    8.4699 │    0.1701 │          8 │   11.2048 │ ✓       │
│ SiluWorkloads │ medium     │ 100 │ 2.48x     │     1.1813 │    0.3976 │    1.04   │    4.2146 │    0.0698 │          5 │    2.9332 │ ✓       │
│ SiluWorkloads │ small      │ 100 │ 1.96x     │     0.4909 │    0.2175 │    0.4407 │    2.6438 │    0.0085 │         16 │    0.9622 │ ✓       │
└───────────────┴────────────┴─────┴───────────┴────────────┴───────────┴───────────┴───────────┴───────────┴────────────┴───────────┴─────────┘

  large: 1.72x faster (95% CI: 6.4302-6.6004ms vs ref 11.2048ms) ✓ significant
  medium: 2.48x faster (95% CI: 1.1034-1.2592ms vs ref 2.9332ms) ✓ significant
  small: 1.96x faster (95% CI: 0.4483-0.5335ms vs ref 0.9622ms) ✓ significant

Kernel: 2385e44  Benchmark: 5b53516
```

## Usage

You must specify which revision to benchmark, either via flags or with `@...` in the repo id:

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

Save results to a file (JSON):

```bash
kernels benchmark kernels-community/activation@v1 --output results.json
```

Benchmark a local kernel checkout (must contain `benchmarks/`):

```bash
kernels benchmark ./my_kernel
```

## Output

- By default, a table is printed (timings in ms).
- `--output <file>.json` writes a JSON payload to disk.

## Writing Benchmark Scripts

Benchmark scripts must live under `benchmarks/` in the kernel repository and match `benchmark*.py`.
Each script should define one or more subclasses of `kernels.benchmark.Benchmark`.

Minimal example (`benchmarks/benchmark_activation.py`):

```python
import torch

from kernels.benchmark import Benchmark


class ActivationBenchmark(Benchmark):
    seed = 0

    def setup(self):
        self.x = torch.randn(128, 1024, device=self.device, dtype=torch.float16)
        self.out = torch.empty(128, 512, device=self.device, dtype=torch.float16)

    def benchmark_silu_and_mul(self):
        self.kernel.silu_and_mul(self.out, self.x)

    def verify_silu_and_mul(self):
        # Return reference tensor; runner compares with self.out
        return torch.nn.functional.silu(self.x[..., :512]) * self.x[..., 512:]
```

The runner will:

- Call `setup()` once per workload (or `setup_<workload>()` if present)
- Warm up (`--warmup`)
- Time `benchmark_<workload>()` for `--iterations`
- If `verify_<workload>()` exists, check that outputs match (`torch.allclose(..., atol=1e-2)`) and show a speedup vs the reference computation

## Troubleshooting

- If the repo does not contain a `benchmarks/` directory (or no `benchmark*.py` files), the command exits with an error.
- If a benchmark script defines no `Benchmark` subclasses, the command exits with an error.
- If `verify_<workload>()` exists and the outputs do not match, the command exits with an error.
