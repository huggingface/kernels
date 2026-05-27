---
name: xpu-kernels
description: "Provides guidance for writing, optimizing, and benchmarking Triton kernels for Intel XPU GPUs (Battlemage/Arc Pro B50) using the Xe-Forge optimization framework. Includes an LLM-driven trial-loop workflow (analyze, validate, benchmark, profile, finalize), XPU-specific patterns (tensor descriptors, GRF mode, tile swizzling), KernelBench fused kernels, and Flash Attention."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: gemm, reduction, flash-attention, optimize, benchmark, tensor-descriptors, xe-forge"
---

# XPU Triton Kernels for Intel GPUs

This skill provides patterns and guidance for developing optimized Triton kernels targeting Intel XPU GPUs (Battlemage/Arc Pro B50). It integrates the [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) optimization framework — an LLM-driven loop that transforms PyTorch code into fast Triton kernels.

## Quick Start

### Optimize a Kernel (Xe-Forge Workflow)

The full optimization workflow analyzes a PyTorch baseline, generates Triton kernel variants in a branching trial tree, benchmarks each on XPU hardware, and finalizes the best result.

```bash
# 1. Analyze the baseline
python scripts/analyze_kernel.py test_kernels/70_Gemm_Sigmoid_Scaling_ResidualAdd_pytorch.py

# 2. Initialize trial tracking
python scripts/trial_manager.py init 70_Gemm_Sigmoid test_kernels/70_Gemm_Sigmoid_Scaling_ResidualAdd_pytorch.py

# 3. Validate a generated kernel (no GPU needed)
python scripts/validate_triton.py my_kernel.py

# 4. Benchmark correctness + performance
python scripts/benchmark.py test_kernels/70_Gemm_Sigmoid_Scaling_ResidualAdd_pytorch.py my_kernel.py

# 5. Profile with VTune (optional)
python scripts/xpu_profiler.py my_kernel.py

# 6. Finalize best trial
python scripts/trial_manager.py finalize 70_Gemm_Sigmoid optimized_triton.py
```

## Supported Hardware

| GPU | Architecture | XVEs | Mem BW | Key Feature | Verified |
|-----|-------------|------|--------|-------------|:--------:|
| **Battlemage G21 / Arc Pro B50** | Xe2 | 128 | ~500 GB/s | Tensor descriptors, GRF 256 | Yes |

> See the [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton) for supported hardware.

## When This Skill Applies

Use this skill when:
- Optimizing PyTorch operations into Triton kernels for **Intel XPU**
- Writing GEMM, fused kernels, reductions, or Flash Attention for Intel GPUs
- Running the **Xe-Forge optimization loop** (analyze → validate → benchmark → profile → finalize)
- Benchmarking kernel performance against PyTorch baseline on XPU

## Xe-Forge Optimization Workflow

Transform PyTorch code into optimized Triton kernels for Intel XPU. Kernels must be numerically equivalent and faster than baseline.

### Configuration — Read `config.yaml` first

At the start of every session, read `scripts/config.yaml`. It controls:
- **`max_trials`** — hard cap on optimization trials; always run all of them (use this instead of hardcoded "10")
- **`vtune_enabled`** — if `false`, skip ALL VTune profiling steps (Step 3.6 and profiler-related decisions)
- **`vtune_bin`** — path to the VTune binary (also settable via `VTUNE_BIN` env var)

### Rules — Never Violate

1. **ONLY create** Triton kernel files (`test_kernels/*_triton.py` or trial files `t<trial_id>.py`).
2. **NEVER create** benchmark scripts, test scripts, helper utilities, or any other Python files.
3. **NEVER write custom scripts** to measure performance or test correctness — ONLY use `scripts/benchmark.py`.
4. If a tool fails, **STOP and report the error**. Do NOT work around it with custom scripts.
5. Generated kernels must be **self-contained** — all helper functions inline.
6. You **MUST run all `max_trials` trials** from `config.yaml`. Do NOT stop early due to plateau — LLM sampling can discover new ideas at any point. The only valid early stop is speedup > 5x.

### Mandatory Tools

**CRITICAL — Single-XPU serialization**: There is only ONE XPU on this machine. You MUST NOT run multiple GPU workloads in parallel. `benchmark.py` and `xpu_profiler.py` must execute strictly one at a time — concurrent GPU jobs produce wrong results. CPU-only tools (`analyze_kernel.py`, `validate_triton.py`, `trial_manager.py`) are safe to parallelize with each other and with anything else.

| Tool | Command | Purpose |
|------|---------|---------|
| **Analyze** | `python scripts/analyze_kernel.py <file>` | Static analysis: operations, shapes, fusion opportunities |
| **Validate** | `python scripts/validate_triton.py <file>` | Syntax + constraint checks before GPU time |
| **Benchmark** | `python scripts/benchmark.py <baseline> <triton> [--triton-baseline] [--baseline-us <cached>]` | Correctness + performance via ai-bench |
| **Profile** | `python scripts/xpu_profiler.py <file>` | VTune GPU hardware counters + recommendations |
| **Init trials** | `python scripts/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]` | Initialize trial tracking |
| **Save trial** | `python scripts/trial_manager.py save <kernel_name> <file> [--parent <parent_id>] [--strategy "..."]` | Save trial to tree |
| **Record result** | `python scripts/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass\|fail> --speedup <float> --baseline_us <float> --triton_us <float>` | Record benchmark result |
| **Check status** | `python scripts/trial_manager.py status <kernel_name>` | View trial tree |
| **Best trial** | `python scripts/trial_manager.py best <kernel_name>` | Get best trial |
| **Baseline time** | `python scripts/trial_manager.py baseline-us <kernel_name>` | Cached baseline time for `--baseline-us` |
| **Finalize** | `python scripts/trial_manager.py finalize <kernel_name> <name>_triton.py` | Copy best trial to output |

### Workflow Steps

#### Step 1: Analyze
- Read the baseline source file. Identify shapes, dtypes, operations, fusion opportunities.
- If baseline is PyTorch: run `python scripts/analyze_kernel.py <pytorch_file>`.
- If baseline is Triton (`--triton-baseline`): skip `analyze_kernel.py` (it only supports PyTorch). Read the Triton file directly.
- Read relevant knowledge base files: start with `references/correctness.yaml` and `references/xpu_optimizations.yaml`.
- Read `references/implementation_reference.md` for templates and the Model class pattern.

#### Step 2: Initialize
```bash
python scripts/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
```

#### Step 3: Trial Loop (always run all `max_trials` from config.yaml)
For each trial:
1. **Write kernel** — start from templates or modify previous trial. See `references/implementation_reference.md`.
2. **Validate** — `python scripts/validate_triton.py <triton_file>` (fix until passing; doesn't count as a trial).
3. **Save** — `python scripts/trial_manager.py save <kernel_name> <triton_file> --parent <parent_id> --strategy "description"`. Omit `--parent` for the first trial (t0).
4. **Benchmark** (MANDATORY every trial):
   - **Trial t0:** `python scripts/benchmark.py <baseline_file> <triton_file> [--triton-baseline]` (measures both baseline and triton).
   - **Trials t1+:** Get cached baseline via `python scripts/trial_manager.py baseline-us <kernel_name>`, then run `python scripts/benchmark.py <baseline_file> <triton_file> [--triton-baseline] --baseline-us <cached_value>` (skips baseline perf, saves time).
   - **After `finalize`:** Re-run `benchmark.py` without `--baseline-us` for final accurate comparison.
5. **Record** — `python scripts/trial_manager.py result <kernel_name> <trial_id> --validation pass --correctness <pass|fail> --speedup <float> --baseline_us <float> --triton_us <float>` (runtimes from benchmark output).
6. **Profile (MANDATORY after t1, if `vtune_enabled` is true in config.yaml)** — Run `python scripts/xpu_profiler.py <triton_file>` after your first benchmarked trial. Use its output to guide subsequent trial strategies. Run again if speedup plateaus after 2+ additional trials. **Skip this step entirely if `vtune_enabled` is false.**
7. **Decide next action** (use profiler output from step 6 to inform decisions):
   - Speedup > 5x → stop (excellent), finalize
   - Speedup improved → continue on this branch, try next optimization level
   - Speedup regressed → branch back to best trial, try different strategy
   - Correctness failed → fix on same branch
   - Profiler says low occupancy (if vtune_enabled) → increase tile sizes, check `references/xpu_optimizations.yaml`
   - Profiler says overhead kernels dominate (if vtune_enabled) → pre-pack to bf16, see `references/optimization_levels.yaml`
   - Plateau → do NOT stop. Try a fundamentally different approach (different algorithm, tiling, fusion strategy). LLM sampling can discover new ideas.
   - See `references/optimization_strategies.md` for the full "try harder" decision tree

#### Step 4: Finalize
```bash
python scripts/trial_manager.py finalize <kernel_name> <name>_triton.py
```

### Reference Docs — Read During Step 1

| Doc | Contents |
|-----|----------|
| `references/implementation_reference.md` | Code templates, Model class pattern, GEMM example |
| `references/optimization_strategies.md` | Strategy reference, optimization levels, checklist |
| `references/workflow_details.md` | Detailed workflow, decision tree, benchmarking/validation details |
| `references/correctness.yaml` | Critical constraints to avoid bugs |
| `references/xpu_optimizations.yaml` | XPU-specific patterns (tensor descriptors, GRF, swizzling) |
| `references/fusion_patterns.yaml` | When to fuse vs split operations |
| `references/optimization_levels.yaml` | Progressive optimization with "try harder" decision tree |

### Existing Baselines Are Naive

The `test_kernels/*.py` Triton files (non-pytorch) are **unoptimized baselines**. They use manual pointer arithmetic, lack autotune, and miss XPU optimizations. Do NOT copy their patterns. Use `references/implementation_reference.md` instead.

## Core XPU Kernel Patterns

### Tensor Descriptors (Preferred on XPU)

Tensor descriptors produce better address generation and memory access codegen than block pointers on Intel XPU.

```python
desc = tl.make_tensor_descriptor(
    base=ptr, shape=[M, N],
    strides=[stride_m, stride_n],
    block_shape=[BLOCK_M, BLOCK_N],
)
block = tl.load(desc, [pid_m, pid_n], boundary_check=(0, 1))
```

### GRF Mode '256'

Use the large register file for compute-heavy kernels:

```python
@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=32)],
    key=['M', 'N', 'K'],
)
@triton.jit(launch_metadata=lambda *args, **kwargs: {'grf_mode': '256'})
def kernel(...):
    ...
```

### Tile Swizzling

Use 1D grid with GROUP_SIZE_M for L2 locality:

```python
grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
# Inside kernel:
pid = tl.program_id(0)
num_pid_n = tl.cdiv(N, BLOCK_N)
group_id = pid // (GROUP_SIZE_M * num_pid_n)
```

### bf16 Inputs with fp32 Accumulation

```python
a = tl.load(a_desc, [pid_m, k], boundary_check=(0, 1))
b = tl.load(b_desc, [k, pid_n], boundary_check=(0, 1))
acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc=acc)  # fp32 accumulator
```

## Critical XPU Constraints

- **NO default values** for `@triton.autotune` meta-parameters in kernel signature
- **1D grid** when using tile swizzling (GROUP_SIZE_M)
- **`boundary_check`** uses dimension indices `(0, 1)`, not booleans
- **Cast batch indices** to `int64` before stride multiplication
- **Prefer tensor descriptors** over block pointers for all new XPU kernels
- **Do NOT mix** block pointer and tensor descriptor APIs on same operation
- **Pre-zero output buffers** when using atomic accumulation
- Model class must be compatible with ai-bench (`nn.Module` with `nn.Linear`)
- Match `get_inputs()`, `get_init_inputs()`, and module-level constants from `*_pytorch.py`

> Full constraint list: [correctness.yaml](references/correctness.yaml)

## Performance Results

Measured on Intel Battlemage G21 / Arc Pro B50 (128 XVEs). All runtimes are median of benchmark trials.

### KernelBench Level 2 — Fused Kernels (bf16)

Speedup is vs. PyTorch eager baseline. Includes GEMM+Sigmoid+Scaling, GEMM+GELU+Softmax, Conv+BatchNorm+ReLU, and other fused patterns.

### Flash Attention Forward (fp16)

Baseline is the flash attention kernel from the Intel XPU Triton backend; speedup is vs. that kernel across multiple sequence lengths.

> Full results: see the [Xe-Forge repository](https://github.com/IntelLabs/Xe-Forge).

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Autotune BLOCK_D** | Wrong results (max_abs 4-8+) | **Never autotune BLOCK_D.** Use `triton.next_power_of_2(D)` |
| Python min/max | Runtime error | `tl.minimum()`/`tl.maximum()` |

## Project Structure

```
xpu-kernels/
├── SKILL.md                                    # This file (skill definition + workflow)
├── manifest.txt                                # Files included in this skill
│
├── scripts/                                    # Standalone CLI tools
│   ├── analyze_kernel.py                       # PyTorch → operations, shapes, fusion opportunities
│   ├── validate_triton.py                      # Syntax + constraint checks
│   ├── benchmark.py                            # Correctness + performance via ai-bench
│   ├── trial_manager.py                        # Tree-structured trial management
│   ├── xpu_profiler.py                         # VTune GPU hardware counters
│   ├── config.py                               # Shared configuration loader
│   ├── config.yaml                             # Session config (max_trials, vtune)
│   └── requirements.txt                        # Python dependencies
│
└── references/                                 # Knowledge base + integration guides
    ├── correctness.yaml                        # Hard constraints for XPU Triton
    ├── xpu_optimizations.yaml                  # Tensor descriptors, GRF, swizzling
    ├── implementation_reference.md             # Code templates, Model class pattern
    ├── implementation_reference.md             # Code templates, Model class pattern
    ├── optimization_strategies.md              # Strategy reference + "try harder" tree
    ├── optimization_levels.yaml                # Progressive L1-L5 optimization levels
    ├── workflow_details.md                     # Detailed workflow and decision tree
    ├── fusion_patterns.yaml                    # When to fuse vs split
    ├── memory_patterns.yaml                    # Access patterns and coalescing
    ├── dtype_optimizations.yaml                # Mixed precision choices
    ├── persistent_kernel_patterns.yaml         # Stream K and persistent kernels
    ├── kernel-templates.md                     # Triton kernel templates for XPU
    └── kernelbench-classification.md           # KernelBench operator taxonomy
```

## See Also

### Xe-Forge Tools
- [analyze_kernel.py](scripts/analyze_kernel.py) — Static analysis of PyTorch reference
- [validate_triton.py](scripts/validate_triton.py) — Pre-benchmark constraint checks
- [benchmark.py](scripts/benchmark.py) — Correctness + performance measurement
- [xpu_profiler.py](scripts/xpu_profiler.py) — VTune GPU hardware counters
- [trial_manager.py](scripts/trial_manager.py) — Branching trial tree management

### XPU Optimization References
- [correctness.yaml](references/correctness.yaml) — Critical constraints
- [xpu_optimizations.yaml](references/xpu_optimizations.yaml) — Tensor descriptors, GRF, swizzling
- [optimization_strategies.md](references/optimization_strategies.md) — Strategy reference
- [optimization_levels.yaml](references/optimization_levels.yaml) — Progressive L1-L5 levels
- [implementation_reference.md](references/implementation_reference.md) — Code templates

### Other References
- [kernelbench-classification.md](references/kernelbench-classification.md) — KernelBench operator taxonomy

### External Resources
- [Xe-Forge Repository](https://github.com/IntelLabs/Xe-Forge)
- [AI-Bench](https://github.com/libxsmm/AI-bench) — Benchmark harness for correctness + performance
- [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)
- [Triton Language Guide](https://triton-lang.org/)
