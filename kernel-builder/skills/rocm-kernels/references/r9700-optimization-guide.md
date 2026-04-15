# R9700 (RDNA4, gfx1201) Optimization Guide

Deep dive into R9700-specific optimizations for Triton kernels on ROCm.

## R9700 RDNA4 Architecture

### Key Specifications

| Component | R9700 | vs MI355X |
|-----------|-------|-----------|
| Compute Capability | gfx1201 | gfx950 |
| Architecture | RDNA4 | CDNA3+ |
| **Wavefront Size** | **32 (Wave32)** | 64 (Wave64) |
| CUs | 64 | 256 |
| Stream Processors | 4096 | - |
| LDS per CU | 64 KB | 160 KB |
| L1 Cache | 32 KB | - |
| L2 Cache | 8 MB | 256 MB |
| L3 Cache | 64 MB | - |
| **Cacheline Size** | **256 B** | - |
| Max Threads/Block | 1024 | 1024 |
| Max Threads/CU | 2048 | 2048 |
| Max Waves/CU | 32 | - |
| SIMDs per CU | 2 | - |
| FP32 Vector TFLOPS | 47.8 | ~200 |
| FP16 Vector TFLOPS | 95.7 | ~2500 |
| FP16 Matrix TFLOPS | 191 | ~2500 |
| Matrix Cores | Limited (no FP8 MFMA) | Full MFMA |

### Critical RDNA4 vs CDNA3+ Differences

1. **Wave32 vs Wave64**: Warp size is 32, same as NVIDIA
2. **No XCD Swizzle**: Single die, no chiplet distribution needed
3. **Limited Matrix Cores**: No FP8 MFMA support
4. **Smaller LDS**: 64 KB vs 160 KB
5. **Smaller L2 Cache**: 8 MB vs 256 MB
6. **256B Cacheline**: Stricter memory alignment requirements
7. **Consumer GPU**: Optimized for inference, not training

## Wave32 Implications

### num_warps Mapping

On RDNA4, `num_warps` still means "number of wavefronts per block":
- 1 warp = 32 threads (Wave32)
- Max 32 waves per CU
- num_warps range: 2-8 (smaller than CDNA)

```python
# CDNA (MI355X): 1 warp = 64 threads
# num_warps=8 → 512 threads/block

# RDNA4 (R9700): 1 warp = 32 threads
# num_warps=8 → 256 threads/block
# Use higher num_warps if needed for same thread count
```

### Reduction Code

Warp-level reductions use different offsets:

```python
# CDNA (Wave64): offsets = 32, 16, 8, 4, 2, 1
# RDNA4 (Wave32): offsets = 16, 8, 4, 2, 1

# In Triton this is handled automatically by tl.sum(), tl.max(), etc.
# No manual shuffle code needed in Triton
```

## Memory Hierarchy

### 256B Cacheline Alignment

R9700 uses 256-byte cachelines (vs 128B on RDNA3). Misaligned accesses are penalized more.

```python
# Ensure contiguous memory access
x = x.contiguous()

# In kernel: sequential access pattern
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets, mask=mask)  # Coalesced
```

### L2 Cache Strategy

With only 8 MB L2, cache reuse is limited:

```python
# For GEMM: use smaller tiles to fit in L2
# BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
# Tile = 64×32×2 + 32×64×2 = 8 KB per stage
# With 2 stages: 16 KB fits in L2 per block
```

### LDS (64 KB) Budget

```
Max LDS per CU = 64 KB

GEMM example (BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, FP16, num_stages=2):
  = (64×32×2 + 32×128×2) × 2 = 24,576 bytes = 24 KB ✓

GEMM example (BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, FP16, num_stages=2):
  = (128×64×2 + 64×128×2) × 2 = 65,536 bytes = 64 KB → Borderline!
```

## Autotune Configurations

### Elementwise Operations

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
```

### GEMM Operations

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8},
                      num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
```

## Grid Sizing

With 64 CUs:

```python
# Aim for multiples of 64 blocks
grid = (triton.cdiv(N, BLOCK_SIZE),)
# For GEMM: grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)
```

## Precision Considerations

- FP16 Matrix TFLOPS = 191 (2x FP32 vector)
- FP16 Vector TFLOPS = 95.7 (2x FP32 vector)
- **No FP8 MFMA**: Cannot use FP8 matrix operations
- INT8 Matrix TOPS = 383 (quantized inference)
- Use FP16 for compute, FP32 for accumulation

## Best Practices Summary

1. **Wave32 awareness**: Use num_warps=2-8
2. **No XCD Swizzle**: Not needed on single-die
3. **Smaller blocks**: 64-128 for GEMM tiles
4. **256B alignment**: Ensure contiguous memory access
5. **LDS budget**: Max 64 KB, keep num_stages=2
6. **Grid sizing**: Multiples of 64 CUs
7. **FP16 preferred**: Best throughput, no FP8 MFMA
8. **L3 cache**: 64 MB can help with model weights
9. **Inference focus**: Best suited for inference workloads
10. **Cacheline**: 256B alignment is stricter than MI355X
