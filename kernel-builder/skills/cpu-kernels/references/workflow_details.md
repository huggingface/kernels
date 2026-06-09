# Detailed Workflow Reference (CPU Kernels)

## Analysis Phase

When given a target PyTorch operation:

1. **Parse the operation** to identify:
   - Input/output shapes and dtypes
   - Mathematical operations (matmul, activations, reductions)
   - Memory access patterns (row-wise, column-wise, random)
   - Compute intensity (FLOPs / bytes transferred)

2. **Consult the knowledge base** (`references/` directory):
   - `runtime_dispatch.yaml`: cpu_features.hpp pattern, dispatch tiers
   - `build_system.yaml`: build.toml multi-target compilation
   - `implementation_reference.md`: C++ templates, Unroll<N>, tinygemm
   - `correctness.yaml`: Critical constraints
   - `simd_optimization_patterns.yaml`: AVX512 vector abstractions
   - `brgemm_patterns.yaml`: brgemm API (GEMM kernels only)

3. **Use the scripts**:
   - `python scripts/analyze_op.py --op <op_name> --shapes <shapes>` — classify kernel type
   - Review `references/implementation_reference.md` for matching template

## Design Phase

1. **Identify the kernel type**: element-wise, reduction, GEMM, attention
2. **Select optimization strategy** from KB:
   - Element-wise: AVX512 vectorization + OpenMP
   - GEMM: tinygemm/brgemm dual-path + parallel_2d
   - Attention: tiled attention + brgemm
3. **Apply critical constraints** (from `references/correctness.yaml`):
   - Always use unaligned loads (`_mm512_loadu_*`)
   - Separate ISA tiers into different translation units
   - Handle tail elements for non-aligned sizes
   - Include `registration.h` in torch_binding.cpp
   - Each kernel has its OWN cpu_features.hpp in its OWN namespace

## Trial Loop Detail

For each trial:

### a. Implement / Modify Kernel
Start from a template (`references/implementation_reference.md`) or modify the previous trial.

### b. Validate
```bash
python scripts/validate_cpu_kernel.py .
```
If validation fails, fix and retry — doesn't count as a new trial.

### c. Build
```bash
kernel-builder build --release
pip install dist/*.whl --force-reinstall
```

### d. Save Trial
```bash
python scripts/trial_manager.py save <kernel_name> <kernel_dir> --parent <parent_id> --strategy "description"
```
For the first trial, omit `--parent`.

### e. Benchmark
```bash
# Trial t0 — measures both baseline and kernel:
python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.forward

# Trials t1+ — use cached baseline to save time:
python scripts/trial_manager.py baseline-us <kernel_name>   # get cached value
python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.forward --baseline-us <cached_value>
```

### f. Record Results
```bash
python scripts/trial_manager.py result <kernel_name> <trial_id> \
    --correctness <pass|fail> --speedup <float> \
    --baseline_us <float> --kernel_us <float>
```

### g. Decision Tree

| Condition | Action |
|-----------|--------|
| **Speedup > `early_stop_speedup`** | Stop — excellent result (the only valid early stop) |
| **Speedup improved** | Continue on this branch, try next optimization |
| **Speedup regressed** | Branch back to best trial, try different strategy |
| **Correctness failed** | Fix on same branch (usually alignment or tail bug) |
| **After t1 (if `perf_stat_enabled`)** | Run `cpu_profiler.py` — mandatory first profile |
| **IPC < 1.0** | Memory bound → add prefetch, reduce tile size |
| **L1 miss > 10%** | Tile too large → reduce tile to fit L1 (48KB) |
| **LLC miss > 20%** | Working set too large → add cache blocking (1MB L2 budget) |
| **Plateau after 2+ trials** | Switch algorithm (tinygemm ↔ brgemm, different blocking) |
| **Max trials reached** | Stop — must run all `max_trials` from `config.yaml` |

### h. Check Status
```bash
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
```

## Trial Manager Commands Reference

```bash
python scripts/trial_manager.py init <kernel_name> <baseline_file>
python scripts/trial_manager.py save <kernel_name> <source> [--parent <parent_id>] [--strategy "..."]
python scripts/trial_manager.py result <kernel_name> <trial_id> [--correctness pass] [--speedup 3.2] [--baseline_us 150.0] [--kernel_us 47.0]
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
python scripts/trial_manager.py baseline-us <kernel_name>
python scripts/trial_manager.py finalize <kernel_name> <output_path>
```

## Benchmarking Details

`scripts/benchmark_cpu.py` uses `torch.utils.benchmark.Timer` for both correctness and performance:

1. **Correctness** — Compares outputs between PyTorch baseline and CPU kernel
   - Uses `torch.allclose()` with atol=1e-2, rtol=1e-2 (bf16 accumulation tolerance)
   - Baseline must define `get_inputs()` and either `get_reference_output()` or `Model` class

2. **Performance** — Benchmarks both implementations on CPU
   - Uses `Timer.blocked_autorange(min_run_time=2.0)` for stable measurements
   - Reports median time and speedup

**Both checks must pass** for the trial to be marked "completed".

## Profiling with perf stat (`scripts/cpu_profiler.py`)

```bash
python scripts/cpu_profiler.py --kernel-package my_kernel --op my_kernel.forward
```

Runs `perf stat` to collect hardware counters and maps bottlenecks to optimization patterns.

### When to Profile
- **MANDATORY** after the first benchmarked trial (t1) — always run at least once
- Run again if speedup plateaus after 2+ additional trials
- When unsure which optimization level to try next

### What It Reports
1. **IPC** (instructions per cycle) — compute vs memory bound indicator
2. **L1 cache miss rate** — tile sizing feedback
3. **LLC (L3) miss rate** — working set size feedback
4. **Branch miss rate** — SIMD vs scalar branching feedback

### How to Use the Output
The profiler prints specific recommendations with references:
```
>> IPC < 1.0: Memory-bound or dependency-bound.
   - Add prefetch instructions (_mm_prefetch with _MM_HINT_T0 or _MM_HINT_T1)
   - Reduce cache blocking tile size
   Reference: references/memory_patterns.yaml
```
Read the referenced file and apply the suggested pattern in your next trial.

## Project Structure

```
cpu-kernels/
├── SKILL.md                            # Core rules and workflow (concise)
├── manifest.txt                        # Files in this skill
│
├── references/                         # Knowledge base
│   ├── implementation_reference.md     # C++ templates, Unroll<N>, tinygemm
│   ├── optimization_strategies.md      # Strategy reference, checklist
│   ├── workflow_details.md             # This file — detailed workflow
│   ├── build_system.yaml               # build.toml multi-target compilation
│   ├── runtime_dispatch.yaml           # cpu_features.hpp + dispatch
│   ├── correctness.yaml                # Critical constraints
│   ├── simd_optimization_patterns.yaml # AVX512 vector abstractions
│   ├── quantized_gemm_patterns.yaml    # LUT + tinygemm/brgemm
│   ├── brgemm_patterns.yaml            # brgemm API, VNNI packing
│   ├── memory_patterns.yaml            # Prefetch, cache blocking
│   ├── threading_patterns.yaml         # OpenMP patterns
│   ├── dtype_optimizations.yaml        # bf16/fp8/int8 handling
│   ├── optimization_levels.yaml        # Progressive L1-L4 checklist
│   └── huggingface-kernels-integration.md # Hub integration
│
└── scripts/                            # Standalone tools (DO NOT recreate)
    ├── config.py                       # Shared config loader
    ├── config.yaml                     # Session config
    ├── analyze_op.py                   # Op analysis → kernel type + strategy
    ├── validate_cpu_kernel.py          # Static checks on C++ kernel code
    ├── benchmark_cpu.py                # Correctness + performance
    ├── cpu_profiler.py                 # perf stat + recommendations
    └── trial_manager.py               # Trial tree management
```
