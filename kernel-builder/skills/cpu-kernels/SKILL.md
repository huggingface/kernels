---
name: cpu-kernels
description: "Provides guidance for writing, optimizing, and benchmarking C++ CPU kernels with SIMD intrinsics (AVX2/AVX512) for the Hugging Face kernels ecosystem. Includes a two-phase workflow: Phase 1 correctness (generic → AVX2) and Phase 2 performance exploration (AVX512 with branching trial loop), runtime CPU dispatch, OpenMP threading, and brgemm integration for GEMM-heavy kernels."
disable-model-invocation: false
user-invocable: true
allowed-tools: "Read, Grep, Glob, Bash"
argument-hint: "kernel type: rmsnorm, flash-attention, quantized-gemm, activation, reduction, optimize, benchmark"
---

# CPU C++ Kernels for x86 Processors

This skill provides patterns and guidance for developing optimized C++ kernels targeting x86 CPUs (Intel Xeon and compatible processors) with AVX2 and AVX512 intrinsics. Kernels are compiled via `kernel-builder` and distributed through the Hugging Face kernels ecosystem.

> **Who runs these commands?** *You*, the agent — not a human. This is an autonomous loop: you write/edit the C++ kernel, build it, then run the scripts below as tools (via Bash) to check correctness, benchmark, and profile. You read each result, record it with `trial_manager.py`, decide the next change from the Phase 2 decision tree, and repeat until you hit `early_stop_speedup` or run all `max_trials`.

## Key Concepts (read before the Quick Start)

The commands use a few names that mean different things. They are **not** interchangeable:

| Name (example) | What it is | Used by |
|----------------|-----------|---------|
| **`baseline.py`** | The **PyTorch reference implementation** you optimize against. It is the ground truth for correctness *and* the speed reference for speedup. **It must define `get_inputs()`** and **either** `get_reference_output()` **or** a `Model` class (plus optional `get_init_inputs()`). You write this file (or it is given) before starting. | every script |
| **`my_rmsnorm`** | A **trial-tree label** — an arbitrary name you pick for this optimization task. `trial_manager.py` stores all attempts under `trials/my_rmsnorm/`. It is *only* a tracking ID. | `trial_manager.py` only |
| **`my_kernel`** | The **installed Python package name** — the build artifact produced by `kernel-builder build` + `pip install`. This is the importable module that contains your compiled kernel. | `--kernel-package` |
| **`my_kernel.rms_norm`** | An **`<package>.<function>` path** — the actual callable inside the installed package. Passed to `--op` to tell the benchmark/profiler which function to run. | `--op` |

> ⚠️ **`--op` means two different things depending on the script.** In `analyze_op.py`, `--op` is a plain **operation name** (e.g. `"rms_norm"`) used to look up compute/memory characteristics. In `benchmark_cpu.py` and `cpu_profiler.py`, `--op` is a **`package.function` path** (e.g. `my_kernel.rms_norm`) used to import and call your kernel. Same flag, different meaning — read each command below carefully.

## Quick Start

### Write a New CPU Kernel

The example below optimizes an RMSNorm kernel. The trial label is `my_rmsnorm`, the built package is `my_kernel`, and its function is `my_kernel.rms_norm` — keep these consistent across all six steps.

```bash
# 1. Analyze the target op. Here --op is an OPERATION NAME (looked up in the
#    knowledge base), not a package path.
python scripts/analyze_op.py --op "rms_norm" --shapes "1024x4096,2048x8192"

# 2. Initialize trial tracking. Args: <trial-label> <baseline-file>.
#    Creates trials/my_rmsnorm/ and records baseline.py as the reference.
python scripts/trial_manager.py init my_rmsnorm baseline.py

# 3. Build the kernel package (produces the installable 'my_kernel' wheel).
cd /path/to/my-kernel && kernel-builder build --release && pip install dist/*.whl --force-reinstall

# 4. Benchmark correctness + performance. Here --op is a PACKAGE.FUNCTION path.
#    Compares my_kernel.rms_norm against baseline.py (correctness + speedup).
python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.rms_norm

# 5. Profile with perf stat (same package.function path as step 4).
python scripts/cpu_profiler.py --kernel-package my_kernel --op my_kernel.rms_norm

# 6. Finalize: promote the best trial in trials/my_rmsnorm/ into output/.
python scripts/trial_manager.py finalize my_rmsnorm output/
```

## Supported Hardware

| ISA | Extensions | Key Instructions | Typical CPUs |
|-----|-----------|-----------------|-------------|
| **AVX2** | FMA, F16C | `_mm256_fmadd_ps`, `_mm256_cvtph_ps` | Most x86 CPUs (2013+) |
| **AVX512** | F, BF16, VL, DQ, BW, VBMI | `_mm512_dpbf16_ps`, `_mm512_permutexvar_epi16` | Intel Xeon |

### GEMM Acceleration: brgemm

For kernels that involve matrix multiplication (quantized GEMM, Flash Attention, MoE), large-M cases use `at::native::cpublas::brgemm()` — a PyTorch wrapper around oneDNN brgemm, which internally dispatches to AMX tile instructions on Intel Xeon (4th Gen+). Small-M cases (M ≤ 4 for bf16) fall back to hand-written `tinygemm` using AVX512 `_mm512_dpbf16_ps`. See [brgemm_patterns.yaml](references/brgemm_patterns.yaml) for details.

> **Note**: brgemm is NOT used in element-wise kernels (RMSNorm, activations, reductions). Those use AVX512 intrinsics directly.

## When This Skill Applies

Use this skill when:
- Writing C++ CPU kernels with SIMD intrinsics for the HF kernels ecosystem
- Optimizing existing CPU kernels (e.g., adding AVX512 to a generic implementation)
- Implementing quantized GEMM kernels (INT4, NF4, FP4, FP8, MXFP4)
- Implementing Flash Attention or other attention kernels for CPU
- Building kernels with `kernel-builder` that target `backend = "cpu"`

## Two-Phase Optimization Workflow

CPU kernel development has two distinct phases with different strategies.

### Configuration — Read `config.yaml` first

At the start of every session, read `scripts/config.yaml`. It controls:
- **`max_trials`** — hard cap on Phase 2 optimization trials
- **`early_stop_speedup`** — speedup vs PyTorch baseline to trigger early stop (default: 3.0)
- **`perf_stat_enabled`** — if `true`, use `perf stat` for profiling (default)
- **`vtune_enabled`** — if `true`, use VTune for detailed microarchitecture analysis
- **`build_command`** — command to build the kernel package

### Rules — Never Violate

1. **ONLY modify** C++ kernel files (`.cpp`, `.hpp`), `torch_binding.cpp`, and `build.toml`. Do NOT create benchmark or test scripts.
2. **NEVER write custom timing code** — ONLY use `scripts/benchmark_cpu.py`.
3. If a tool fails, **STOP and report the error**. Do NOT work around it with custom scripts.
4. Generated kernels must follow the **runtime dispatch pattern** with `cpu_features.hpp` — see `references/runtime_dispatch.yaml`.
5. Every kernel should have a **generic ATen fallback** that works on any CPU. If a specific path cannot have a meaningful fallback, use `TORCH_CHECK(false, ...)` with a clear error message.
6. Each SIMD tier (AVX2, AVX512) must be in a **separate translation unit** (`.cpp` file) with its own compiler flags in `build.toml`. Do NOT mix intrinsics from different ISA levels in the same file.
7. All SIMD implementations must handle **edge cases** (hidden_size not divisible by vector width).
8. AVX2 tier is **optional** — most CPU kernels go directly from generic fallback to AVX512. Only add AVX2 when it provides meaningful benefit for element-wise ops.
9. You **MUST run all `max_trials` trials** in Phase 2. Do NOT stop early due to plateau — the only valid early stop is speedup > `early_stop_speedup`.

### Mandatory Tools

| Tool | Command | Purpose |
|------|---------|---------|
| **Analyze** | `python scripts/analyze_op.py --op <op_name> --shapes <shapes>` | Analyze PyTorch op: compute/memory characteristics, SIMD strategy recommendations |
| **Validate** | `python scripts/validate_cpu_kernel.py <kernel_dir>` | Static checks: alignment, OpenMP usage, intrinsics correctness, build.toml validation |
| **Build** | `kernel-builder build --release` | Compile C++ kernel via build.toml into a wheel |
| **Benchmark** | `python scripts/benchmark_cpu.py <baseline_file> --kernel-package <pkg> --op <func>` | Correctness + performance via `torch.utils.benchmark` |
| **Profile** | `python scripts/cpu_profiler.py --kernel-package <pkg> --op <func>` | `perf stat` hardware counters + optimization recommendations |
| **Trial Manager** | `python scripts/trial_manager.py <command> ...` | Trial tree management (init/save/result/status/best/finalize) |

> **Benchmark discipline**: Pin to a single NUMA node — `numactl --cpunodebind=0 --membind=0 python scripts/benchmark_cpu.py ...`. See [threading_patterns.yaml](references/threading_patterns.yaml).


### Phase 1: Correctness (Linear, No Branching)

Build the kernel tier by tier. Each tier must be correct before moving on.

#### Tier 0: Generic Fallback
- Implement using **PyTorch ATen ops only** (no intrinsics).
- This serves as the portable baseline that runs on any CPU.
- Must produce results matching the PyTorch reference within tolerance.
- File: `<kernel>_cpu/<kernel>_cpu.cpp`

```bash
# Validate + build
python scripts/validate_cpu_kernel.py .
kernel-builder build --release
pip install dist/*.whl --force-reinstall

# Benchmark (this also establishes the PyTorch baseline time)
python scripts/benchmark_cpu.py baseline.py --kernel-package my_kernel --op my_kernel.rms_norm
```

#### Tier 1: AVX2 (Optional)
- Add AVX2 implementation using `_mm256_*` intrinsics.
- Compile with `-mavx2 -mfma -mf16c -fopenmp`.
- Must be correct; performance improvement is a bonus.
- File: `<kernel>_cpu/<kernel>_avx2.cpp`

#### Tier 2: AVX512
- Add AVX512 implementation using `_mm512_*` intrinsics.
- Compile with `-mavx512f -mavx512bf16 -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mfma -mf16c -fopenmp`.
- For GEMM kernels using brgemm, additionally add `-mamx-tile -mamx-bf16 -mamx-int8`.
- This is the **entry point to Phase 2** — performance optimization starts here.
- File: `<kernel>_cpu/<kernel>_avx512.cpp`

### Phase 2: Performance Exploration (Branching, With Backtracking)

Once AVX512 is correct, optimize for peak performance. This phase uses the trial manager with branching.

```bash
# Initialize Phase 2 trials
python scripts/trial_manager.py init <kernel_name> baseline.py

# For each trial:
# 1. Modify kernel code (AVX512 intrinsics, or brgemm for GEMM kernels)
# 2. Build
kernel-builder build --release && pip install dist/*.whl --force-reinstall
# 3. Benchmark
python scripts/benchmark_cpu.py baseline.py --kernel-package <pkg> --op <func>
# 4. Save trial
python scripts/trial_manager.py save <kernel_name> <dir> --parent <parent_id> --strategy "description"
# 5. Record result
python scripts/trial_manager.py result <kernel_name> <trial_id> --correctness pass --speedup <float> --baseline_us <float> --kernel_us <float>
# 6. Profile (after t1, or when plateaued)
python scripts/cpu_profiler.py --kernel-package <pkg> --op <func>
```

#### Phase 2 Decision Tree

| Condition | Action |
|-----------|--------|
| **Speedup > `early_stop_speedup`** | Stop — excellent result (the only valid early stop) |
| **Speedup improved** | Continue on this branch, try next optimization |
| **Speedup regressed** | Branch back to best trial, try different strategy |
| **Correctness failed** | Fix on same branch (usually alignment or SIMD boundary bug) |
| **After t1 (if `perf_stat_enabled`)** | Run `cpu_profiler.py` — mandatory first profile |
| **IPC < 1.0** | Memory bound → add prefetch, change cache blocking |
| **L1 miss rate high** | Tile too large for L1 → reduce tile size |
| **L3 miss rate high** | Working set too large → add cache blocking |
| **Plateau after 2+ trials** | Do NOT keep tuning the same knobs. Change the **approach**: switch algorithm path (tinygemm ↔ brgemm), change the fusion/blocking/data-layout strategy, or reconsider the dispatch heuristic. A different structure beats endless parameter sweeps. |
| **Max trials reached** | Stop — must run all `max_trials` from `config.yaml` |

#### Optimization Search Space (Phase 2)

> These tables are a **starting menu of values seen in existing kernels, not an exhaustive recipe**. Use them to seed trials, but when a branch plateaus, prefer a structurally different idea (algorithm, fusion, memory strategy) over sweeping these knobs further. See the try-harder tree in [optimization_levels.yaml](references/optimization_levels.yaml).

**GEMM kernels (quantized GEMM, Flash Attention, MoE):**

| Dimension | Actual Values in Existing Kernels | Notes |
|-----------|----------------------------------|-------|
| BLOCK_M (tinygemm path) | 4 | Small M, fused dequant+GEMM |
| BLOCK_M (brgemm path) | 32 (= 2×TILE_M) | Must be multiple of TILE_M=16 |
| BLOCK_N (tinygemm) | 32, 64 | Determines register tile COLS = BLOCK_N/16 |
| BLOCK_N (brgemm) | 32 (= 2×TILE_N) | Must be multiple of TILE_N=16 |
| BLOCK_K | 128 (= 4×TILE_K) | K-dimension blocking |
| BLOCK_M/N (flash-attn2) | 256 / 768 | Much larger — attention-specific |
| K-loop unroll | 4 (`#pragma GCC unroll 4`) | All GEMM kernels use 4 |
| Prefetch distance | 0 (disabled), 64 elements ahead | L1 prefetch via `_MM_HINT_T0` |
| Algorithm path | `use_brgemm` threshold (e.g. M > 4) | Switch from tinygemm to brgemm |
| brgemm dequant policy | `use_brgemm_dequant_out` (e.g. M > 100)| True=pre-dequant all B upfront; False=dequant per K-block |
| L2 cache budget | 1 MB (50% of 2 MB L2) | Controls N-blocking in `loop_2d` |
| Thread decomposition | 2D factorization: nth_m × nth_n | Based on M/N aspect ratio |

**Element-wise kernels (RMSNorm, activations):**

| Dimension | Actual Values | Notes |
|-----------|---------------|-------|
| Vectorization width | 16 (fp32), 32 (bf16) | Per-type VEC_ELEM_NUM |
| Prefetch hint | `_MM_HINT_T1` (L2) | Different from GEMM (L1) |
| OpenMP grain size | 1024 | `at::parallel_for` grain |
| Threading | `#pragma omp parallel for` over rows | Simple 1D parallelism |

### Phase 2 Finalization

```bash
python scripts/trial_manager.py finalize <kernel_name> output/
# Re-run benchmark without cached baseline for final accurate comparison
python scripts/benchmark_cpu.py baseline.py --kernel-package <pkg> --op <func>
```

## Reference Docs — Read During Phase 1

| Doc | Contents |
|-----|----------|
| `references/runtime_dispatch.yaml` | cpu_features.hpp pattern, dispatch tiers |
| `references/build_system.yaml` | build.toml multi-target CPU compilation |
| `references/implementation_reference.md` | C++ kernel templates, Unroll\<N\>, tinygemm, torch_binding.cpp |
| `references/correctness.yaml` | Critical constraints: alignment, FTZ/DAZ, denormals |

## Reference Docs — Read During Phase 2

| Doc | Contents |
|-----|----------|
| `references/simd_optimization_patterns.yaml` | AVX2/AVX512 vector abstractions and patterns |
| `references/quantized_gemm_patterns.yaml` | LUT + tinygemm + Unroll template for 4-bit GEMM |
| `references/brgemm_patterns.yaml` | brgemm API usage, VNNI packing, tinygemm vs brgemm selection (GEMM kernels only) |
| `references/memory_patterns.yaml` | Prefetch, alignment, cache blocking |
| `references/threading_patterns.yaml` | OpenMP parallel patterns |
| `references/dtype_optimizations.yaml` | bf16/fp8/int8 handling and conversion on CPU |
| `references/optimization_levels.yaml` | Progressive L1→L5 optimization checklist + try-harder tree |
| `references/optimization_strategies.md` | Strategy reference, decision tree, checklist |
| `references/workflow_details.md` | Detailed trial loop workflow |
| `references/huggingface-kernels-integration.md` | Hub integration for CPU kernels |

## Core CPU Kernel Patterns

### Runtime Dispatch (Required for All Kernels)

Every CPU kernel has its own `cpu_features.hpp` (in its own namespace) and dispatches at runtime. Most kernels dispatch as AVX512 → fallback (no AVX2 tier):

```cpp
// my_kernel_cpu/cpu_features.hpp — each kernel has its OWN copy
namespace my_kernel_cpu {
class CPUFeatures {
public:
    static bool hasAVX512BF16() { /* CPUID + XCR0 checks */ }
    static bool hasAVX2() { /* CPUID check */ }
    // GEMM kernels also check: static bool hasAMX() { ... }
};
}

// my_kernel_cpu/my_kernel_cpu.cpp — dispatcher
#include "cpu_features.hpp"
#include "my_kernel_avx512.hpp"

void my_kernel(torch::Tensor& out, const torch::Tensor& input, ...) {
    if (CPUFeatures::hasAVX512BF16()) {
        avx512::my_kernel_impl(out, input, ...);
    } else {
        // ATen fallback — inline or in a separate _fallback.cpp
        out = torch::some_aten_op(input, ...);
    }
}
```

> **Note**: Only rmsnorm has a three-tier dispatch (AVX512 → AVX2 → ATen). GEMM kernels skip AVX2. Flash-attn2 additionally requires AMX via `hasAllRequiredFeatures()`.

> Full pattern: [runtime_dispatch.yaml](references/runtime_dispatch.yaml)

### build.toml Multi-Target Compilation

Each SIMD tier is a separate `[kernel.*]` section with its own compiler flags. The `include` directive is required for header resolution:

```toml
[kernel.my_kernel_cpu]
backend = "cpu"
depends = ["torch"]
include = ["my_kernel_cpu"]
src = [
    "my_kernel_cpu/my_kernel_cpu.cpp",
    "my_kernel_cpu/my_kernel_cpu_torch.cpp",
    "my_kernel_cpu/my_kernel_cpu.hpp",
    "my_kernel_cpu/cpu_features.hpp",
]

[kernel.my_kernel_cpu_avx512]
backend = "cpu"
# Note: For GEMM kernels (e.g., flash-attn2, megablocks), you must also include "-mamx-tile", "-mamx-bf16", "-mamx-int8"
cxx-flags = ["-mavx512f", "-mavx512bf16", "-mavx512vl", "-mavx512dq", "-mavx512bw", "-mavx512vbmi", "-mfma", "-mf16c", "-fopenmp"]
depends = ["torch"]
include = ["my_kernel_cpu"]
src = [
    "my_kernel_cpu/my_kernel_avx512.cpp",
    "my_kernel_cpu/my_kernel_avx512.hpp",
]
```

> **Note**: Every section needs `include = ["<kernel_dir>"]` for header resolution. The `_torch.cpp` file bridges Python-facing declarations to the C++ dispatcher. AVX2 section is optional (only rmsnorm has one).

> Full pattern: [build_system.yaml](references/build_system.yaml)

### torch_binding.cpp Registration

All kernels use `registration.h` macros for op registration:

```cpp
#include "registration.h"

// Forward declarations
#if defined(CPU_KERNEL)
torch::Tensor my_kernel_cpu_forward(torch::Tensor input, torch::Tensor weight, float eps);
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("forward(Tensor input, Tensor weight, float eps) -> Tensor");
    ops.impl("forward", torch::kCPU, &my_kernel_cpu_forward);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

> **Note**: `registration.h` is provided by kernel-builder. Multi-device kernels (rmsnorm, megablocks) use `#if defined(CPU_KERNEL)` / `#elif defined(CUDA_KERNEL)` guards.

### Vector Type Abstractions (AVX512)

Wrap raw intrinsics in typed vector classes for readability:

```cpp
// cpu_types_avx512.hpp
struct FP32Vec16 {
    __m512 reg;
    FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}
    FP32Vec16(__m512 r) : reg(r) {}
    FP32Vec16 operator*(const FP32Vec16& other) const {
        return FP32Vec16(_mm512_mul_ps(reg, other.reg));
    }
    float reduce_sum() const { return _mm512_reduce_add_ps(reg); }
};
```

> Full pattern: [simd_optimization_patterns.yaml](references/simd_optimization_patterns.yaml)

### Quantized GEMM Template (INT4/NF4/FP4)

All 4-bit quantized GEMM kernels share the same skeleton — only the LUT and zero-point handling differ:

```
nibble split → zero subtract → LUT lookup → _mm512_dpbf16_ps accumulate → scale fmadd (per group) → bf16 output
```

The parameterized components:
- **LUT**: GPTQ (linear INT4), BnB (NF4/FP4), MegaBlocks (FP8/MXFP4)
- **Zero-point**: per-group (GPTQ), none/encoded in LUT (BnB), per-block (FP8)
- **Algorithm**: tinygemm (small M, fused) vs brgemm (large M, unpack+BLAS)
- **Weight conversion**: The C++ kernel expects a specific block-interleaved format, NOT raw checkpoint format. Each framework converts in its own repo:
  - **GPTQ**: `transform_cpu()` unpacks int32→uint8, reorders by g_idx, transposes to [N,K]; then `convert_weight_packed_zp()` repacks to [N,K/2] block-interleaved (BLOCK_N=32). Zeros unpacked to [groups,N] uint8. Scales to bf16. Done at first forward in GPTQModel repo.
  - **BnB**: `_convert_weight_packed_for_cpu()` unpacks uint8 nibbles→[N,K], repacks to [N,K/2] block-interleaved (same algo as GPTQ). Denests nested absmax. Transposes scales to [K/blocksize,N] bf16. Done at first forward in bitsandbytes repo.
  - **Megablocks MoE**: `ops.convert_weight_packed()` does transpose+VNNI pack. `ops.convert_scale_packed()` reorders scales. Cached via `packed_weight=True`.
- **VNNI Conversion (K/V Activations)**:
  - **Flash Attention**: `pack_vnni()` per tile per forward (K/V change every call, so caching is not possible).
- **Element-wise (RMSNorm)**: No conversion needed.

> Full pattern: [quantized_gemm_patterns.yaml](references/quantized_gemm_patterns.yaml), weight conversion: [brgemm_patterns.yaml](references/brgemm_patterns.yaml)

## Critical CPU Constraints

- **Always use unaligned loads**: All existing kernels use `_mm512_loadu_*` exclusively. Never use `_mm512_load_*`.
- **Edge cases**: When `hidden_size % VEC_ELEM_NUM != 0`, handle the tail with scalar or masked SIMD ops.
- **FTZ/DAZ**: Flush-to-zero and denormals-as-zero may be set by PyTorch. Do NOT assume IEEE 754 denormal behavior.
- **OpenMP overhead**: For small tensors, use `adjust_num_threads(m)` to reduce thread count. GEMM kernels use `parallel_2d` for 2D thread decomposition.
- **bf16 precision**: `_mm512_dpbf16_ps` accumulates in fp32 but inputs are bf16 — precision loss is expected. Use atol=1e-2 for correctness checks.
- **Data alignment**: Use `alignas(64)` for stack-allocated tile buffers to optimize cache-line access.

> Full constraint list: [correctness.yaml](references/correctness.yaml)

## Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| **Unaligned access** | SEGFAULT or wrong results | Use `_mm512_loadu_*` instead of `_mm512_load_*` |
| **Missing tail handling** | Wrong results for non-aligned sizes | Add scalar loop for remainder elements |
| **OpenMP on small tensor** | Slower than baseline | Add `if (num_tokens > threshold)` guard |
| **Wrong compiler flags** | Intrinsics not recognized | Check build.toml `cxx-flags` matches code |
| **Silent scalar `at::vec`** | Kernel ~2x slow, no error; `objdump` shows 0 `%zmm` / `nm` shows `expf@GLIBC` | Define `CPU_CAPABILITY_AVX512` for TUs using `at::vec::Vectorized` (see build_system.yaml) |
| **CPUID detection wrong** | Crashes on older CPU | Verify `cpu_features.hpp` checks OS support (XCR0) |

## Project Structure

```
cpu-kernels/
├── SKILL.md                                    # This file (skill definition + workflow)
├── manifest.txt                                # Files included in this skill
│
├── scripts/                                    # Standalone CLI tools
│   ├── analyze_op.py                           # PyTorch op → compute/memory analysis
│   ├── validate_cpu_kernel.py                  # Static checks on C++ kernel code
│   ├── benchmark_cpu.py                        # Correctness + performance via torch.utils.benchmark
│   ├── cpu_profiler.py                         # perf stat hardware counters + recommendations
│   ├── trial_manager.py                        # Tree-structured trial management
│   ├── config.yaml                             # Session config (max_trials, profiler, build)
│   └── config.py                               # Shared configuration loader
│
└── references/                                 # Knowledge base
    ├── correctness.yaml                        # Critical constraints for CPU kernels
    ├── runtime_dispatch.yaml                   # cpu_features.hpp + dispatch pattern
    ├── build_system.yaml                       # build.toml multi-target CPU compilation
    ├── simd_optimization_patterns.yaml         # AVX2/AVX512 vector abstractions and patterns
    ├── quantized_gemm_patterns.yaml            # LUT + tinygemm/brgemm template
    ├── brgemm_patterns.yaml                     # brgemm API, VNNI packing, tinygemm fallback (GEMM kernels only)
    ├── memory_patterns.yaml                    # Prefetch, alignment, cache blocking
    ├── threading_patterns.yaml                 # OpenMP parallel patterns
    ├── dtype_optimizations.yaml                # bf16/fp8/int8 handling on CPU
    ├── optimization_levels.yaml                # Progressive L1→L5 optimization checklist
    ├── implementation_reference.md             # C++ kernel templates and examples
    ├── optimization_strategies.md              # Strategy reference + decision tree
    ├── workflow_details.md                     # Detailed workflow reference
    └── huggingface-kernels-integration.md      # HF kernels ecosystem integration guide
```

## See Also

### Tools
- [analyze_op.py](scripts/analyze_op.py) — Analyze PyTorch op characteristics
- [validate_cpu_kernel.py](scripts/validate_cpu_kernel.py) — Static kernel validation
- [benchmark_cpu.py](scripts/benchmark_cpu.py) — Correctness + performance measurement
- [cpu_profiler.py](scripts/cpu_profiler.py) — perf stat hardware counters
- [trial_manager.py](scripts/trial_manager.py) — Trial tree management

### CPU Optimization References
- [correctness.yaml](references/correctness.yaml) — Critical constraints
- [simd_optimization_patterns.yaml](references/simd_optimization_patterns.yaml) — SIMD patterns
- [quantized_gemm_patterns.yaml](references/quantized_gemm_patterns.yaml) — Quantized GEMM template
- [optimization_levels.yaml](references/optimization_levels.yaml) — Progressive optimization
- [implementation_reference.md](references/implementation_reference.md) — Code templates

### External Resources
- [Hugging Face Kernels](https://github.com/huggingface/kernels) — Kernel hub and builder CLI
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [kernel-builder Documentation](https://github.com/huggingface/kernels/tree/main/docs)
- [xpu-kernels skill](../xpu-kernels/SKILL.md) — the Intel XPU Triton skill this workflow was adapted from
- [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) — the LLM-driven optimization framework the skill methodology originates from

## Acknowledgments

The methodology of this skill — the YAML knowledge base, the benchmark/validation harnesses, and the branching trial-manager optimization loop — was adapted from the [xpu-kernels skill](../xpu-kernels/SKILL.md) built by a group of Intel AI researchers, the IntelLabs team behind [Xe-Forge](https://github.com/IntelLabs/Xe-Forge), where the methodology originates. Thanks to the original authors for a solid foundation to build on.
