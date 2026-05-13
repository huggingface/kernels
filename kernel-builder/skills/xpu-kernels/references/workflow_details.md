# Detailed Workflow Reference

## Analysis Phase

When given a PyTorch kernel (typically `*_pytorch.py`, but can be any user-specified path):

> **Note**: The existing `test_kernels/*.py` Triton files (non-pytorch) are **naive, unoptimized baselines**. Do NOT treat them as examples of good Triton code. Use `references/implementation_reference.md` and `references/examples/` instead.

1. **Parse the PyTorch code** to identify:
   - Input/output shapes and dtypes
   - Mathematical operations (matmul, activations, reductions)
   - Operation fusion opportunities
   - Memory access patterns

2. **Consult the knowledge base** (`references/` directory):
   - `xpu_optimizations.yaml`: XPU-specific patterns (tensor descriptors, GRF mode, warp count, tile swizzling)
   - `fusion_patterns.yaml`: When to fuse operations
   - `memory_patterns.yaml`: Memory access best practices
   - `correctness.yaml`: Critical constraints to avoid bugs
   - `dtype_optimizations.yaml`: Data type choices

3. **Use the skills** to help:
   - `python scripts/analyze_kernel.py <pytorch_file>` - Extract operation structure
   - Review `references/examples/index.yaml` for similar patterns

## Design Phase

1. **Identify the kernel type**: Pure GEMM, GEMM + epilogue, GEMM + reduction, complex fusion
2. **Select optimization strategies** from KB (memory, tiling, parallelism, fusion, dtypes)
3. **Apply critical constraints** (from `references/correctness.yaml` and `references/xpu_optimizations.yaml`):
   - NO default values for `@triton.autotune` meta-parameters in kernel signature
   - Use 1D grid when applying tile swizzling (GROUP_SIZE_M)
   - boundary_check uses dimension indices (0, 1), not booleans
   - Cast batch indices to int64 before stride multiplication
   - Do NOT mix block pointer and tensor descriptor APIs on same operation
   - Pre-zero output buffers when using atomic accumulation
   - Model class must be compatible with ai-bench (standard `nn.Module` with `nn.Linear`)

## Trial Loop Detail

For each trial:

### a. Implement / Modify Kernel
Start from a template (`references/implementation_reference.md`) or modify the previous trial's code. See `references/implementation_reference.md`.

### b. Validate Syntax
```bash
python scripts/validate_triton.py <triton_file>
```
If validation fails, fix and retry - doesn't count as a new trial. Note: `<triton_file>` should be `t<trial_id>.py`.

### c. Save Trial
```bash
python scripts/trial_manager.py save <kernel_name> <triton_file> --parent <parent_id> --strategy "description"
```
For the first trial, omit `--parent`.

### d. Benchmark
```bash
# Trial t0 — measures both baseline and triton:
python scripts/benchmark.py <baseline_file> <triton_file> [--triton-baseline]

# Trials t1+ — use cached baseline to save time:
python scripts/trial_manager.py baseline-us <kernel_name>   # get cached value
python scripts/benchmark.py <baseline_file> <triton_file> [--triton-baseline] --baseline-us <cached_value>

# After finalize — re-run without --baseline-us for final accurate comparison
```

### e. Record Results
```bash
python scripts/trial_manager.py result <kernel_name> <trial_id> \
    --validation pass --correctness <pass|fail> --speedup <float> \
    --baseline_us <float> --triton_us <float>
```

### f. Decision Tree

| Condition                                  | Action                                                                                                                                                |
|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Speedup > 5x**                           | Stop - excellent result (the only valid early stop)                                                                                                   |
| **Speedup improved**                       | Continue on this branch, try next optimization level                                                                                                  |
| **Speedup regressed**                      | Branch back to best trial, try a different strategy                                                                                                   |
| **Correctness failed**                     | Fix code on same branch                                                                                                                               |
| **After t1 (if `vtune_enabled`)**          | Run `python scripts/xpu_profiler.py <triton_file>` — mandatory first profile                                                                           |
| **Speedup plateaued after 2+ more trials** | Run profiler again (if `vtune_enabled`); try a fundamentally different approach                                                                       |
| **Plateau / diminishing returns**          | Do NOT stop. Try a fundamentally different approach (different algorithm, tiling, fusion strategy). LLM sampling can discover new ideas at any point. |
| **Max trials reached**                     | Stop — must run all `max_trials` from `config.yaml`                                                                                                   |

### g. Check Status
```bash
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
```

## Trial Manager Commands Reference
```bash
python scripts/trial_manager.py init <kernel_name> <baseline_file> [--triton-baseline]
python scripts/trial_manager.py save <kernel_name> <file> [--parent <parent_id>] [--strategy "..."]
python scripts/trial_manager.py result <kernel_name> <trial_id> [--validation pass] [--correctness pass] [--speedup 3.2] [--baseline_us 150.0] [--triton_us 47.0]
python scripts/trial_manager.py status <kernel_name>
python scripts/trial_manager.py best <kernel_name>
python scripts/trial_manager.py baseline-us <kernel_name>
python scripts/trial_manager.py finalize <kernel_name> <name>_triton.py
```

## Benchmarking Details

`scripts/benchmark.py` uses ai-bench (`modules/ai-bench/`) for both correctness and performance:

1. **Correctness** - Compares outputs between PyTorch and Triton implementations
   - Uses `check_correctness()` with per-variant tolerances from YAML spec (defaults: rtol=1e-2, atol=1e-5)
   - Syncs model weights via `copy_model_weights()`
   - Falls back to direct module loading when no spec file is available

2. **Performance** - Benchmarks both implementations on XPU hardware
   - Reads the YAML spec file (auto-detected from `modules/ai-bench/problems/specs/KernelBench/level*/`)
   - Reports speedup metrics (Triton vs PyTorch) per spec variant

**Both checks must pass** for the kernel to be considered complete.

**Setup**: External tools must be initialised: `git submodule update --init`

## Profiling with VTune (`scripts/xpu_profiler.py`)

```bash
python scripts/xpu_profiler.py <triton_file> [--warmup 5] [--iters 20]
```

Runs Intel VTune `gpu-offload` collection to capture both Level Zero API tasks and OA (Observation Architecture) hardware counters, then maps bottlenecks to KB optimization patterns.

**Prerequisite**: OA counters require `observation_paranoid=0`:
```bash
echo 0 | sudo tee /proc/sys/dev/xe/observation_paranoid
```

### When to Profile
- **MANDATORY** after the first benchmarked trial (t1) — always run at least once per session
- Run again if speedup plateaus after 2+ additional trials
- You're unsure which optimization level to try next

### What It Reports
1. **Platform info**: GPU name, XVE count, max frequency
2. **Host tasks**: CPU-side overhead (JIT compilation, data copies, synchronization)
3. **GPU computing tasks table** (per-kernel): Time, instance count, XVE Active/Stalled/Idle %, occupancy %, memory bandwidth read/write
4. **Primary kernel detail**: Full OA hardware counter breakdown including:
   - XVE execution: Active/Stalled/Idle percentages
   - Occupancy limiters: Work Size Limit, SLM Use Limit, Barriers Use Limit (tells WHY occupancy is low)
   - Memory bandwidth: Read/Write GB/s
   - Cache hierarchy: L3 Busy/Stalled %, L3 Miss Ratio, LSC Miss Ratio, LSC→L3 Miss Ratio
   - Register spill size, SLM bank conflicts, TLB misses
5. **Optimization recommendations**: Each grounded in a specific KB pattern:
   - XVE Stalled > Active → memory bound → `references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern)` + `references/optimization_levels.yaml (level_2)`
   - Low occupancy + Work Size limiter → grid too small → `references/xpu_optimizations.yaml (xpu_tile_swizzling)`
   - Low occupancy + SLM limiter → tile too large → `references/xpu_optimizations.yaml (xpu_grf_mode)`
   - High L3 Miss → poor reuse → `references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling)`
   - Register spill > 0 → reduce liveness → `references/memory_patterns.yaml (reduce_liveness_sink_load_and_prefetch)`
   - Overhead kernels dominate → pre-pack to bf16 → `references/optimization_levels.yaml (level_2_bandwidth_reduction)`
   - Host time >> GPU time → sync in hot path → `references/memory_patterns.yaml (no_device_to_host_scalar_sync)`

### How to Use the Output
The profiler prints specific recommendations with references:
```
>> XVE Stalled (72%) > Active (28%): memory/dependency bound.
   Use tensor descriptors for better address codegen, pre-pack to bf16 to halve bandwidth.
   Reference: references/xpu_optimizations.yaml (xpu_descriptor_gemm_pattern, xpu_tile_swizzling) +
              references/optimization_levels.yaml (level_2_bandwidth_reduction)
```
Read the referenced file and apply the suggested pattern in your next trial.

## Validation Details

`scripts/validate_triton.py` checks:
- Syntax correctness
- Autotune config issues (no default params)
- Grid/swizzling consistency
- boundary_check format
- Data type usage

## Project Structure
```
xpu-kernels/
├── SKILL.md                    # Core rules and workflow (concise)
│
├── references/                      # Knowledge base
│   ├── implementation_reference.md  # Templates, code patterns, Model class
│   ├── optimization_strategies.md   # Strategy reference, checklist, KB index
│   ├── workflow_details.md          # This file — detailed workflow
│   ├── correctness.yaml             # Correctness constraints
│   ├── xpu_optimizations.yaml       # XPU-specific patterns
│   ├── optimization_levels.yaml     # Progressive optimization checklist
│   ├── fusion_patterns.yaml         # Kernel fusion guidelines
│   ├── memory_patterns.yaml         # Memory access optimizations
│   ├── dtype_optimizations.yaml     # Data type optimizations
│   └── persistent_kernel_patterns.yaml # Stream K and persistent kernel patterns
│
└── scripts/                         # Standalone tools (DO NOT recreate)
    ├── analyze_kernel.py            # PyTorch → operations, shapes, fusion opportunities
    ├── validate_triton.py           # Syntax + constraint checks before benchmarking
    ├── benchmark.py                 # Correctness + performance via ai-bench
    ├── trial_manager.py             # Tree-structured trial init/save/record/finalize
    ├── xpu_profiler.py              # VTune GPU hardware counters + recommendations
    ├── config.yaml                  # max_trials, vtune_enabled, vtune_bin
    └── config.py                    # Shared configuration loader for config.yaml
```
