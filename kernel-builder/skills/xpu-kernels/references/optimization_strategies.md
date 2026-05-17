# Optimization Strategies Reference

## Optimization Levels (Iterative Deepening)

| Level | Focus | Typical Speedup |
|-------|-------|-----------------|
| **1. Baseline XPU** | Tensor descriptors, tile swizzling, `@triton.autotune`, fused epilogue | 1.5-3x |
| **2. Bandwidth** | Pre-pack to bf16, grf_mode='256', `tl.dot(a, b, acc=acc)` | 2-4x |
| **3. Algebraic** | Fold BN/scale/affine into weights (eliminate epilogue) | 3-6x |
| **4. Expert** | Stream K, persistent kernels, warp sweeping | 5-10x+ |

**"Try harder" decision tree** (from `references/optimization_levels.yaml`):
- Speedup < 2x after Level 1 -> apply Level 2 (bandwidth is the bottleneck)
- Speedup 2-3x after Level 2 -> check Level 3 (can epilogue be algebraically eliminated?)
- Speedup 3-5x -> good for most workloads; Level 4 only for critical-path kernels
- Speedup > 5x -> diminishing returns, stop

**Case study**: Kernel #39 (Gemm_Scale_BatchNorm) went from 2.69x (Level 1) to 5.28x (Level 2+3) by pre-packing to bf16 and folding BN into GEMM weights.

## GEMM Kernels
1. Use tensor descriptors (preferred on XPU) or block pointers (not manual pointer arithmetic)
2. Apply tile swizzling with GROUP_SIZE_M (1D grid required)
3. `@triton.autotune` with varied configs - sweep block sizes, warps, GRF mode
4. Large tiles for square matrices: 256x256, 32 warps, grf_mode='256'
5. Smaller tiles for skinny-M: BLOCK_M in {32, 64}, fewer warps
6. Mixed precision: bf16/fp16 inputs, fp32 accumulator
7. Pre-pack weight transposes: `weight_t = weight.t().contiguous()` once in `_pack_weights()`
8. Pre-pack to bf16: Convert weights AND inputs to bf16 before kernel launch (not in-kernel) - see `references/dtype_optimizations.yaml`
9. Algebraic weight folding: Fold BN/scale/affine into GEMM weights at pack time - see `references/fusion_patterns.yaml`

## Fusion
1. Fuse light epilogues: bias + simple activation (ReLU, SiLU)
2. Be cautious with heavy chains: multiple exp/tanh/clamp can hurt register pressure
3. Split GEMM + reduction: Use 2D GEMM -> separate reduction kernel (don't serialize over N)

## Reductions (Softmax, LayerNorm)
1. Multi-row tiling: Process multiple rows per program (BLOCK_SIZE_Y)
2. Query hardware limits: Use `max_work_group_size` to compute BLOCK_SIZE_Y
3. Power-of-2 blocks: `BLOCK_SIZE_X = triton.next_power_of_2(n_cols)`
4. Sweep warp_size: Try both 16 and 32 with different num_warps

## Critical "DO NOT" List
- Do NOT put default values on `@triton.autotune` meta-parameters in kernel signature
- Do NOT use 2D grid with tile swizzling (must be 1D)
- Do NOT repack weights inside forward() hot path
- Do NOT implement GEMM2 by looping all N tiles inside one program
- Do NOT mix block pointer and tensor descriptor APIs on same load/store
- Do NOT use fp64 unless absolutely required (5-10x slower)

## KB Quick Index

- **Starting a GEMM kernel?** -> `references/xpu_optimizations.yaml`
- **Fusing operations?** -> `references/fusion_patterns.yaml`
- **Memory access issues?** -> `references/memory_patterns.yaml`
- **Kernel crashes or wrong results?** -> `references/correctness.yaml`
- **Slow due to fp64?** -> `references/dtype_optimizations.yaml`
- **Advanced techniques?** -> `references/persistent_kernel_patterns.yaml`
- **Need more speedup?** -> `references/optimization_levels.yaml`
- **Looking for examples?** -> `references/examples/index.yaml` + `references/examples/*.py`

## Common Patterns Checklist

When transforming PyTorch -> Triton:

- [ ] Identified operation type (GEMM, reduction, elementwise)
- [ ] Chosen memory access pattern (tensor descriptors preferred; block pointers as fallback)
- [ ] Applied tile swizzling (if GEMM)
- [ ] `@triton.autotune` with varied BLOCK_M/N/K, num_warps, grf_mode configs
- [ ] NO default values on autotune meta-parameters in kernel signature
- [ ] Used 1D grid if swizzling
- [ ] Mixed precision: bf16/fp16 -> fp32 accumulator
- [ ] Fused light epilogues only
- [ ] Pre-packed weight transposes (cached in `_pack_weights()`)
- [ ] Model class compatible with ai-bench (standard nn.Module with nn.Linear)
- [ ] Matched `get_inputs()`, `get_init_inputs()`, module-level constants from *_pytorch.py
- [ ] Triton file name matches base kernel name (for spec YAML auto-detection)
- [ ] Validated with `python scripts/validate_triton.py <triton_file>`
- [ ] Benchmarked with `python scripts/benchmark.py <pytorch_file> <triton_file>`
