# Autotune Guide

## What @triton.autotune Does

At the first call with a given set of `key` values, Triton compiles every config,
benchmarks each one, picks the fastest, and caches the result. Subsequent calls
with the same key values use the cached winner instantly. When key values change,
it re-benchmarks.

## Config Structure

```python
triton.Config(
    kwargs={'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},
    num_stages=3,
    num_warps=8,
)
```

- **kwargs**: compile-time constants passed to the kernel
- **num_warps**: number of warps (groups of 32 threads on NVIDIA, 64 on AMD) per program
- **num_stages**: software pipelining depth

## The key Parameter

```python
@triton.autotune(configs=[...], key=['M', 'N', 'K'])
```

`key` lists the runtime arguments whose values determine which config is fastest.
When any key value changes, the autotuner re-runs.

Typical choices:
- Matmul: `key=['M', 'N', 'K']`
- Element-wise: `key=['n_elements']`
- Row reduction: `key=['n_cols']`

## When NOT to Autotune

Never autotune a parameter that controls the reduction dimension width:

```python
# DANGEROUS — if autotuner picks BLOCK_D = 64 but D = 2048, only first 64
# elements of each row are processed. Results are silently wrong.
@triton.autotune(configs=[
    triton.Config({'BLOCK_D': 64}),
    triton.Config({'BLOCK_D': 128}),
    triton.Config({'BLOCK_D': 256}),
], key=['M'])
def rmsnorm_kernel(..., BLOCK_D: tl.constexpr):
    ...
```

Instead, compute it in the wrapper:

```python
def rmsnorm(x, weight, eps=1e-6):
    M, D = x.shape
    BLOCK_D = triton.next_power_of_2(D)  # always covers full row
    rmsnorm_kernel[(M,)](x, weight, out, D, BLOCK_D=BLOCK_D)
```

Safe things to autotune:
- BLOCK_M, BLOCK_N for matmul output tile (programs handle independent tiles)
- BLOCK_K for the K-loop step size (you loop over the full K regardless)
- num_warps, num_stages

## num_warps Selection

| Tile size | Recommended num_warps |
| --- | --- |
| <= 1024 elements | 2-4 |
| 1024-4096 elements | 4-8 |
| >= 4096 elements | 8-16 |

More warps = more threads = better for larger tiles. But more warps also means
more register pressure per program, which can reduce occupancy (programs per SM).

## num_stages Selection

| Workload type | Recommended num_stages |
| --- | --- |
| Memory-bound (softmax, norm, activation) | 2 |
| Compute-bound (matmul) | 3-5 |
| Large SRAM available (>200KB/SM) | 4 |
| Small SRAM (<100KB/SM) | 2 |

More stages means more tiles "in flight" (being prefetched while current one
computes), but each stage uses SRAM to hold a tile. Too many stages and you
run out of SRAM.

## Grid as Lambda

When block sizes are autotuned, the grid can't be computed before the config is
selected. Use a lambda:

```python
grid = lambda meta: (
    triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
)
autotuned_kernel[grid](a, b, c, M, N, K)
```

## Practical Config Sets

### Matmul (good starting point)

```python
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
]
```

### Element-wise with varying sizes

```python
configs = [
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
]
```

## Debugging Autotune

Set `TRITON_PRINT_AUTOTUNING=1` environment variable to see which config wins
and the benchmark times for each.

```bash
TRITON_PRINT_AUTOTUNING=1 python your_script.py
```
