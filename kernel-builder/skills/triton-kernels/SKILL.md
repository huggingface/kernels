| name | triton-kernels |
| --- | --- |
| description | Provides guidance for writing and benchmarking portable Triton kernels targeting NVIDIA and AMD GPUs. Covers core DSL patterns, @triton.autotune, numerics (fp16/bf16/fp8), masked loads, reductions, tiling, benchmarking harness, correctness testing, and integration with HuggingFace Kernels Hub (get_kernel). Vendor-neutral: points to rocm-kernels and xpu-kernels for backend-specific tuning. |
| disable-model-invocation | false |
| user-invocable | true |
| allowed-tools | Read, Grep, Glob, Bash |
| argument-hint | kernel type: softmax, matmul, rmsnorm, layernorm, activation, reduction, element-wise, autotune, benchmark, correctness, get_kernel, transformers, diffusers |

# Portable Triton Kernels

This skill provides patterns and guidance for developing portable, optimized
Triton kernels that run on NVIDIA and AMD GPUs without modification. For
backend-specific tuning, see [rocm-kernels](../rocm-kernels/SKILL.md) (AMD) and
[xpu-kernels](../xpu-kernels/SKILL.md) (Intel).

## When This Skill Applies

Use this skill when:

- Writing new Triton kernels for normalization, activation, attention, or linear algebra ops
- Deciding block sizes, num_warps, num_stages, and autotune configs
- Handling numerics (fp32 accumulation, bf16/fp16 input/output, masked values)
- Setting up correctness tests against a PyTorch reference
- Benchmarking kernel throughput (GB/s or TFLOPS)
- Publishing a Triton kernel to the HuggingFace Kernels Hub via get_kernel
- Fusing multiple ops into a single kernel to reduce DRAM round-trips

## Hard Constraints

1. **BLOCK_SIZE for reductions must cover the full reduction dimension.** Use
   `triton.next_power_of_2(dim)` in the Python wrapper. Never autotune BLOCK_SIZE
   when it controls the reduction axis — partial rows give wrong results silently.

2. **Masked loads need a safe `other` value.** Use `other=0.0` for additive
   contexts (sum, dot product). Use `other=float('-inf')` for max-based reductions
   (softmax numerator). Using the wrong fill value is the #1 cause of subtle
   numerical bugs.

3. **Accumulate in fp32.** Cast inputs to `tl.float32` before reductions. Cast
   back to the input dtype only at the final `tl.store`. Half-precision
   accumulation compounds rounding errors across hundreds of additions.

4. **All tensors must be contiguous.** Assert `x.is_contiguous()` in the Python
   wrapper, or call `.contiguous()` before computing pointers. Non-contiguous
   tensors break flat-offset pointer arithmetic.

5. **tl.constexpr parameters must be known at compile time.** BLOCK_SIZE,
   num_warps, num_stages are compile-time constants. Pass them as kernel
   arguments with the `: tl.constexpr` annotation or via autotune configs.

## Core DSL Patterns

### Program IDs and Grid Launch

Every Triton kernel is launched as a grid of programs. Each program gets a unique
ID via `tl.program_id(axis)`.

```python
@triton.jit
def my_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # ... compute ...
    tl.store(output_ptr + offsets, result, mask=mask)
```

Grid sizing:
```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
my_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
```

For 2D grids (e.g. matmul):
```python
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
# program_id(0) = row block, program_id(1) = col block
```

### Masked Loads and Stores

Always mask when BLOCK_SIZE may exceed the actual dimension:

```python
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_cols
x = tl.load(ptr + offsets, mask=mask, other=0.0)
tl.store(out_ptr + offsets, result, mask=mask)
```

### 2D Pointer Arithmetic (Tiling)

For loading 2D tiles (matmul, attention):

```python
row_offsets = row_start * BLOCK_M + tl.arange(0, BLOCK_M)
col_offsets = col_start * BLOCK_N + tl.arange(0, BLOCK_N)

# Broadcasting: [:, None] makes column vector, [None, :] makes row vector
ptrs = base_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :]
mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
tile = tl.load(ptrs, mask=mask, other=0.0)
```

### Reductions

Row-wise reduction (softmax, layernorm, rmsnorm):

```python
# One program per row. BLOCK_SIZE >= n_cols (next power of 2).
pid = tl.program_id(0)
row_start = pid * stride_row
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < n_cols

x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0).to(tl.float32)
row_sum = tl.sum(x, axis=0)
row_max = tl.max(x, axis=0)
```

### The dot Product (Tile-Level Matmul)

`tl.dot(a, b)` performs a tile-level matrix multiply that maps to tensor cores.
Requires both operands to have their K dimension >= 16.

```python
# a_tile: (BLOCK_M, BLOCK_K), b_tile: (BLOCK_K, BLOCK_N)
acc += tl.dot(a_tile, b_tile)  # acc: (BLOCK_M, BLOCK_N)
```

## Autotune

`@triton.autotune` benchmarks multiple kernel configurations at runtime and
caches the fastest one per problem shape.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # re-tune when these values change
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

Guidelines:

- **key=** lists the runtime values that affect which config is fastest. Usually
  the matrix dimensions.
- **num_warps**: More warps = more threads per program. Large tiles (128x256) need
  8-16 warps. Small tiles (32x64) need 2-4.
- **num_stages**: Software pipelining. With N stages, N-1 tiles are being
  prefetched while 1 is being computed. More stages need more SRAM. Use 2-3 for
  memory-bound ops, 3-5 for compute-bound.
- **When NOT to autotune**: Never autotune a parameter that controls the reduction
  dimension (BLOCK_D for rmsnorm, BLOCK_SIZE for softmax row width). The kernel
  silently produces wrong results if the autotuner picks a value smaller than the
  actual dimension.
- **Grid as lambda**: When block sizes are autotuned, use a lambda for the grid
  since dimensions aren't known until a config is selected:
  ```python
  grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
  ```

### num_warps Heuristic

```python
num_warps = 4
if BLOCK_SIZE >= 2048:
    num_warps = 8
if BLOCK_SIZE >= 4096:
    num_warps = 16
```

### num_stages Heuristic

```python
# Query SRAM per SM if available, otherwise default to 2
num_stages = 4 if sram_per_sm > 200_000 else 2
```

## Numerics

### fp32 Accumulation (Critical)

```python
# Load in native dtype, cast to fp32 for compute, cast back for store
x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
# ... reductions and arithmetic in fp32 ...
tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
```

Without fp32 accumulation, a sum over 512+ elements in fp16 accumulates visible
rounding error (max_abs_diff > 1e-2 vs torch reference).

### Safe Softmax Pattern

```python
x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf')).to(tl.float32)
x_max = tl.max(x, axis=0)
x = x - x_max                  # shift for numerical stability
numerator = tl.exp(x)          # exp(-inf) = 0, harmless
denominator = tl.sum(numerator, axis=0)
result = numerator / denominator
```

The `other=float('-inf')` ensures masked positions don't affect the max or the
sum (since exp(-inf) = 0).

### Dtype Handling

```python
# Detect input dtype and cast back at the end
input_dtype = x.dtype  # only works outside @triton.jit; inside, use tl.float16 etc.

# Inside kernel: always accumulate in fp32
x_fp32 = x.to(tl.float32)
# ... compute ...
out = result.to(x_fp32.dtype)  # won't work — use explicit dtype
out = result.to(tl.bfloat16)   # explicit cast
```

## Reference Kernel: Softmax

Complete fused softmax — one DRAM read, one DRAM write per row:

```python
@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(output_ptr + row_start + offsets, result.to(tl.float16), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else (8 if BLOCK_SIZE <= 4096 else 16)
    softmax_kernel[(n_rows,)](
        x, output, n_cols, x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return output
```

Why this beats PyTorch: naive softmax does ~8MN memory ops (max, subtract, exp,
sum, divide — each round-trips through DRAM). This kernel does 2MN (one read,
one write). ~4x reduction in memory traffic.

## Reference Kernel: Matmul with Tiling

Blocked matrix multiplication with accumulation loop:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn,
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c,
             mask=(rm[:, None] < M) & (rn[None, :] < N))
```

Key points:
- Accumulate in fp32, store in target dtype
- `other=0.0` for masked positions — zeros don't affect addition
- `tl.dot` requires K dimension >= 16

## Reference Kernel: RMSNorm

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_row, D,
    eps: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    x = tl.load(x_ptr + row * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)

    if HAS_WEIGHT:
        w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        out = x * rms_inv * w
    else:
        out = x * rms_inv

    tl.store(out_ptr + row * stride_row + offsets, out.to(x.dtype), mask=mask)


def triton_rmsnorm(x, weight=None, eps=1e-6):
    assert x.is_contiguous()
    x_2d = x.view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    has_weight = weight is not None
    if not has_weight:
        weight = torch.empty(0, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_kernel[(M,)](
        x_2d, weight, out, x_2d.stride(0), D, eps, has_weight,
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)
```

## Benchmarking

### Throughput Measurement

Use `triton.testing.do_bench` for accurate GPU timing:

```python
ms = triton.testing.do_bench(lambda: my_kernel_wrapper(x))
```

### GB/s Calculation

For memory-bound kernels (softmax, rmsnorm, activations):

```python
# Total bytes moved = bytes_read + bytes_written
# For in-place-like ops: 2 * numel * element_size (one read + one write)
gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
```

### TFLOPS Calculation

For compute-bound kernels (matmul):

```python
# Matmul: 2*M*N*K FLOPs (multiply + add for each output element)
tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
```

### Benchmark Harness Template

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='kernel-performance',
        args={'M': 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: my_softmax(x))
    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps

benchmark.run(save_path='.', print_data=True)
```

### Correctness Testing

Always validate against PyTorch before benchmarking:

```python
def test_kernel(size, atol=1e-3, rtol=1e-3):
    torch.manual_seed(0)
    x = torch.randn(size, device='cuda')
    y_triton = my_kernel(x)
    y_ref = torch_reference(x)
    torch.testing.assert_close(y_triton, y_ref, atol=atol, rtol=rtol)
    print("PASSED")

# Test with irregular shapes to verify masking
test_kernel((1823, 781))
test_kernel((1, 1))
test_kernel((4096, 4096))
```

## HuggingFace Kernels Hub Integration

Load pre-compiled kernels from the Hub (no local compilation):

```python
from kernels import get_kernel, has_kernel

# Check availability and load
if has_kernel("kernels-community/activation", version=1):
    activation = get_kernel("kernels-community/activation", version=1)
    y = torch.empty_like(x)
    activation.gelu_fast(y, x)
```

For Triton kernels specifically, the kernel is packaged as a noarch build (no
per-CUDA-version compilation needed). The `build.toml` uses `backends = ["triton"]`.

### Publishing Your Own Triton Kernel

```toml
# build.toml
[general]
name = "my-triton-kernel"
backends = ["triton"]
version = 1
license = "Apache-2.0"

[general.hub]
repo-id = "your-username/my-triton-kernel"
```

Build and upload:
```bash
kernel-builder build-and-upload
```

## Transformers Integration

Patch model modules with your custom kernel:

```python
from transformers import AutoModelForCausalLM

def patch_rmsnorm(model):
    for name, module in model.named_modules():
        if type(module).__name__ == 'RMSNorm':
            eps = getattr(module, 'variance_epsilon', None) or getattr(module, 'eps', 1e-6)
            has_weight = hasattr(module, 'weight') and module.weight is not None
            if has_weight:
                def make_forward(mod, epsilon):
                    def forward(x):
                        return triton_rmsnorm(x, mod.weight, eps=epsilon)
                    return forward
                module.forward = make_forward(module, eps)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16)
model.to("cuda")
patch_rmsnorm(model)
```

Key points:
- Use `type(module).__name__` not `isinstance()` (catches all RMSNorm variants)
- Weight may be None if `elementwise_affine=False`
- Inject before `enable_model_cpu_offload()`

## Common Pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| Silent wrong results | Autotuned BLOCK_SIZE < reduction dim | Compute BLOCK_SIZE = triton.next_power_of_2(dim) in wrapper, never autotune |
| NaN / huge values | fp16 accumulation overflow | Cast to tl.float32 before reductions |
| Wrong softmax output | other=0.0 in masked load | Use other=float('-inf') for max-reduction inputs |
| Crash on non-contiguous input | Flat offset math assumes contiguity | Assert is_contiguous() or call .contiguous() |
| "K >= 16" error from tl.dot | BLOCK_K too small | Minimum BLOCK_K = 16 (tensor core constraint) |
| Autotune runs every call | Missing key= parameter | Add key=['M', 'N', 'K'] to cache across calls |
| Performance worse than PyTorch at small N | Kernel launch overhead dominates | Triton kernels shine at large N; small N is launch-bound |

## Performance Profiling

```bash
# NVIDIA
nsys profile -o profile python your_kernel.py
ncu --set full -o metrics python your_kernel.py

# AMD (ROCm) — see rocm-kernels skill
rocprof --stats python your_kernel.py
```

## See Also

### Scripts

- [benchmark_template.py](scripts/benchmark_template.py) — Reusable benchmark harness for any Triton kernel
- [correctness_template.py](scripts/correctness_template.py) — Correctness test template

### References

- [autotune-guide.md](references/autotune-guide.md) — Deep dive on autotune configs, key parameter, and pitfalls
- [kernel-patterns.md](references/kernel-patterns.md) — Additional kernel patterns (dropout, fused add+norm, element-wise)
- [benchmarking-guide.md](references/benchmarking-guide.md) — GB/s vs TFLOPS, roofline analysis, when to expect gains

### Backend-Specific Skills

- [rocm-kernels](../rocm-kernels/SKILL.md) — AMD MI355X / R9700 tuning, ROCm constraints
- [xpu-kernels](../xpu-kernels/SKILL.md) — Intel XPU Triton tuning

### External Resources

- [Triton Documentation](https://triton-lang.org/)
- [HuggingFace Kernels Documentation](https://huggingface.co/docs/kernels/)
- [Community Kernels on Hub](https://huggingface.co/kernels-community)
