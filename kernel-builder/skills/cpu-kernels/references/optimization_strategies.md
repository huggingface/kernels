# Optimization Strategies Reference (CPU)

## Optimization Levels (Iterative Deepening)

| Level | Focus | Typical Speedup |
|-------|-------|-----------------|
| **L1. Baseline AVX512** | Correct vectorization, unaligned loads, OpenMP threading | 1.5-3x |
| **L2. Memory** | Prefetch (L1/L2), cache blocking, streaming stores | 2-4x |
| **L3. Compute** | FMA utilization, loop unrolling, brgemm for GEMM | 3-6x |
| **L4. Expert** | 2D thread decomposition, tinygemm micro-kernel, VNNI packing | 5-10x+ |

## Decision Tree

- Speedup < 2x after L1 → apply L2 (memory is the bottleneck — add prefetch)
- Speedup 2-3x after L2 → check L3 (are FMA instructions being generated?)
- Speedup 3-5x → good for most workloads; L4 only for critical-path GEMM kernels
- Speedup > `early_stop_speedup` → stop (excellent result)

## Element-wise Kernels (RMSNorm, Activations)

1. Use `_mm512_loadu_ps` / `_mm512_storeu_ps` for all memory access
2. Handle tail elements with scalar loop or masked operations
3. Use `_mm512_fmadd_ps` for multiply-add (saves one instruction per FMA)
4. Thread over rows with `#pragma omp parallel for schedule(static)`
5. Prefetch next row data with `_mm_prefetch(ptr, _MM_HINT_T1)` (L2)
6. Use vector type abstractions (FP32Vec16, BF16Vec32) for readability
7. OpenMP grain size: 1024 for `at::parallel_for`

## GEMM Kernels (Quantized GEMM, MoE)

1. Implement dual-path: tinygemm (M ≤ 4) + brgemm (M > 4)
2. Use `Unroll<N>` template for compile-time loop unrolling
3. Use `_mm512_dpbf16_ps` for bf16 dot-product accumulation in tinygemm
4. Use `at::native::cpublas::brgemm()` for large-M GEMM
5. VNNI packing for brgemm inputs: interleave bf16 pairs for AMX
6. 2D thread decomposition via `parallel_2d(m, n, fn)`
7. K-loop unroll factor: 4 (`#pragma GCC unroll 4`)
8. L2 cache budget: 1MB (50% of 2MB L2) controls N-blocking

## Attention Kernels (Flash-Attention)

1. Tile attention: BLOCK_M=256, BLOCK_N=768
2. Use brgemm for Q@K^T and Softmax(S)@V matmul blocks
3. Online softmax with running max/sum for numerical stability
4. Thread over batch × heads × M-tiles
5. Requires AVX512 + AMX (via brgemm)

## Critical "DO NOT" List

- Do NOT use aligned loads (`_mm512_load_*`) — always use `_mm512_loadu_*`
- Do NOT mix AVX2 and AVX512 intrinsics in the same file
- Do NOT call AMX instructions directly — use brgemm wrapper only
- Do NOT use `double` (float64) in hot paths — use float/bf16
- Do NOT forget tail handling for non-aligned hidden sizes
- Do NOT skip the ATen fallback tier

## Profiling Quick Reference

| Metric | Threshold | Action |
|--------|-----------|--------|
| IPC < 1.0 | Memory bound | Add prefetch, reduce tile size |
| L1 miss > 10% | Working set too large | Reduce blocking to fit L1 (48KB) |
| LLC miss > 20% | Data doesn't fit L3 | Add cache blocking for L2 (1MB budget) |
| Branch miss > 5% | Unpredictable branches | Use SIMD masking, `__builtin_expect` |

## KB Quick Index

- **Starting a kernel?** → `references/implementation_reference.md`
- **Build system?** → `references/build_system.yaml`
- **Runtime dispatch?** → `references/runtime_dispatch.yaml`
- **GEMM kernel?** → `references/brgemm_patterns.yaml` + `references/quantized_gemm_patterns.yaml`
- **SIMD patterns?** → `references/simd_optimization_patterns.yaml`
- **Memory issues?** → `references/memory_patterns.yaml`
- **Threading?** → `references/threading_patterns.yaml`
- **Wrong results?** → `references/correctness.yaml`
- **Data types?** → `references/dtype_optimizations.yaml`
- **Need more speedup?** → `references/optimization_levels.yaml`

## Common Patterns Checklist

When building a CPU kernel:

- [ ] Identified operation type (element-wise, reduction, GEMM, attention)
- [ ] Created `cpu_features.hpp` with own namespace
- [ ] Implemented ATen fallback in dispatcher
- [ ] AVX512 implementation with proper compiler flags
- [ ] Tail handling for non-aligned sizes
- [ ] `torch_binding.cpp` with registration.h macros
- [ ] `build.toml` with `include` directive in every section
- [ ] Validated with `python scripts/validate_cpu_kernel.py .`
- [ ] Benchmarked with `python scripts/benchmark_cpu.py`
- [ ] Profiled with `python scripts/cpu_profiler.py` (after first correct trial)
