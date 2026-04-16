# ROCm Triton Kernel Troubleshooting Guide

Common issues and solutions when developing Triton kernels for AMD GPUs.

## Build / Import Issues

### 1. `tl.libdevice` Not Found

**Error:** `AttributeError: module 'triton.language' has no attribute 'libdevice'`

**Cause:** `tl.libdevice` is CUDA-only (NVIDIA's libdevice library).

**Fix:** Replace with manual implementations:
```python
# WRONG (CUDA only)
tl.libdevice.tanh(x)
tl.libdevice.log1p(x)

# CORRECT (ROCm compatible)
e2x = tl.exp(2.0 * x); tanh_x = (e2x - 1.0) / (e2x + 1.0)
log1p_x = tl.log(1.0 + x)
```

### 2. `tl.tanh` / `tl.math.tanh` Not Available

**Error:** `AttributeError: module 'triton.language.math' has no attribute 'tanh'`

**Cause:** Neither `tl.tanh`, `tl.math.tanh`, nor `tl.libdevice.tanh` exist on ROCm Triton. This is the most common GEGLU compilation failure.

**Fix — manual tanh (ONLY reliable method):**
```python
x_f32 = x.to(tl.float32)
e2x = tl.exp(2.0 * x_f32)
tanh_x = (e2x - 1.0) / (e2x + 1.0)
```

## Runtime Errors

### 3. HIP Runtime Error: Invalid Argument

**Error:** `hipErrorInvalidValue` or `HIP Error: invalid argument`

**Common causes:**
- Grid/block size exceeds hardware limits
- Mismatched tensor shapes
- LDS overflow

**Fix:**
```python
# Check grid size
grid = (triton.cdiv(N, BLOCK_SIZE),)
assert grid[0] > 0, f"Grid size must be > 0, got {grid[0]}"

# Ensure contiguous tensors
x = x.contiguous()

# Reduce num_stages to avoid LDS overflow
# num_stages=2 is safest
```

### 4. HIP Out of Memory (LDS)

**Error:** `AMDGPU_KERNEL_ERROR_OUT_OF_MEMORY` or `LDS size exceeds limit`

**Cause:** Kernel uses more LDS than available (64 KB on R9700, 160 KB on MI355X).

**Fix:**
```python
# Reduce num_stages
num_stages=2  # instead of 3 or 4

# Reduce block sizes
BLOCK_M=64, BLOCK_N=64, BLOCK_K=32  # smaller tiles
```

### 5. Kernel Timeout

**Error:** Kernel hangs or times out.

**Common cause:** Grid and Program ID mismatch.

```python
# WRONG: 1D grid but 2D program_id
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
pid_m = tl.program_id(0)  # OK
pid_n = tl.program_id(1)  # ERROR: axis 1 doesn't exist in 1D grid

# CORRECT: Compute 2D indices from 1D grid
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
pid = tl.program_id(0)
pid_m = pid // triton.cdiv(N, BLOCK_N)
pid_n = pid % triton.cdiv(N, BLOCK_N)
```

## Correctness Issues

### 6. Autotuning BLOCK_D Causes Wrong Results

**Symptom:** RMSNorm/AdaLN/GEGLU correctness fails with large `max_abs` errors (4-8+). Kernel runs fast but produces garbage.

**Cause:** `@triton.autotune` with `BLOCK_D` configs (e.g., 512, 1024, 2048, 4096) may select a `BLOCK_D < D` (hidden dimension). Since `tl.arange(0, BLOCK_D)` only covers `BLOCK_D` elements, the kernel processes a partial row, computing wrong variance and writing incomplete output.

**Fix:** Never autotune `BLOCK_D` for row-reduction kernels. Compute it dynamically:
```python
# WRONG — autotune may pick BLOCK_D=512 when D=2048
@triton.autotune(configs=[
    triton.Config({'BLOCK_D': 512}, num_warps=4),
    triton.Config({'BLOCK_D': 1024}, num_warps=8),
], key=['D'])

# CORRECT — compute in Python wrapper
BLOCK_D = triton.next_power_of_2(D)
num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
kernel[(M,)](..., BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2)
```

### 7. RoPE cos/sin Out-of-Bounds GPU Crash (batch > 1)

**Symptom:** `Memory access fault by GPU node` crash. Only happens when batch_size > 1.

**Cause:** cos/sin tensors have shape `[seq_len, head_dim]`, but when the grid is `(batch * seq_len, num_heads)`, `pid_s` ranges `[0, batch * seq_len)`. For `pid_s >= seq_len`, `cos_ptr + pid_s * head_dim` is out of bounds.

**Fix:** Use modular indexing for cos/sin:
```python
# WRONG — crashes when pid_s >= seq_len
cos_val = tl.load(cos_ptr + pid_s * head_dim + offs, ...)

# CORRECT — wrap position index for batch dimension
seq_idx = pid_s % seq_len
cos_val = tl.load(cos_ptr + seq_idx * head_dim + offs, ...)
```

### 8. FP16/BF16 Precision Loss

**Symptom:** Results differ from PyTorch reference by more than tolerance.

**Fix:** Always accumulate in FP32:
```python
# WRONG: Accumulate in FP16
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

# CORRECT: Accumulate in FP32, cast at store
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... computation ...
result = acc.to(tl.float16)
tl.store(out_ptr + ..., result, mask=mask)
```

**Tolerance guidelines:**
- BF16 (7-bit mantissa): `atol=0.1`, `rtol=1e-2`
- FP16 (10-bit mantissa): `atol=0.01`, `rtol=1e-3`

### 9. Mask Errors

**Error:** `ValueError: Mask argument cannot be block type`

**Fix:** Ensure mask dimensions match pointer dimensions:
```python
# 1D kernel
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

# 2D kernel
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
x = tl.load(ptr + offs_m[:, None] * stride + offs_n[None, :], mask=mask, other=0.0)
```

### 10. Python min/max Inside Kernel

**Error:** `TypeError` or incorrect results.

**Fix:**
```python
# WRONG: Python builtins
result = min(a, b)
result = max(a, b)

# CORRECT: Triton functions
result = tl.minimum(a, b)
result = tl.maximum(a, b)
```

## Performance Issues

### 11. GEMM Extremely Slow (0.3-0.5x)

**Cause:** Missing XCD swizzle on MI355X.

**Fix:** Add XCD swizzle pattern (see Template 5: GEMM with XCD Swizzle in kernel-templates.md).

### 12. Elementwise Kernel Slow

**Common causes:**
1. BLOCK_SIZE too small → not utilizing bandwidth
2. Internal loops → should process full block
3. Missing autotune → not finding optimal config

**Fix:**
```python
# Use large BLOCK_SIZE for elementwise
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
    ],
    key=['n_elements'],
)
```

### 13. Missing @triton.autotune (for elementwise)

**Symptom:** Kernel runs but performance is poor.

**Fix:** **EVERY kernel must have autotune with 4+ configs.** Fixed block sizes are almost never optimal.

### 14. tl.store() Keyword Argument Error

**Error:** `TypeError: store() got an unexpected keyword argument`

**Fix:** Check Triton version API. Use positional arguments if needed:
```python
# Check your Triton version
# tl.store(ptr, value, mask=mask)  # Most versions
# tl.store(ptr, value, mask)       # Some older versions
```

### 15. eps: tl.constexpr Causes Recompilation Crash

**Error:** `AttributeError("'NoneType' object has no attribute 'type'")` during Triton compilation

**Cause:** When `eps` is declared as `tl.constexpr`, the kernel is compiled separately for each unique eps value. If the kernel first compiles with `eps=1e-6` and later is called with `eps=1e-8` (e.g., from `nn.RMSNorm.eps`), the recompilation on ROCm Triton can crash.

**Fix:** Remove `tl.constexpr` from `eps` and pass it as a regular runtime parameter:
```python
# WRONG — triggers recompilation for each eps value, may crash on ROCm
@triton.jit
def rmsnorm_kernel(x_ptr, ..., eps: tl.constexpr, BLOCK_D: tl.constexpr):
    ...

# CORRECT — eps is a regular runtime float, no recompilation
@triton.jit
def rmsnorm_kernel(x_ptr, ..., eps, BLOCK_D: tl.constexpr):
    ...

# Also ensure eps is a plain float in the wrapper
rmsnorm_kernel[(M,)](..., float(eps), BLOCK_D=BLOCK_D, ...)
```

**Note:** Only `BLOCK_D`, `HAS_WEIGHT`, and other values that change kernel structure should be `tl.constexpr`. Parameters like `eps` that only affect numerical values should be regular parameters.

## Debugging Tips

### Check GPU Architecture

```bash
rocminfo | grep "Name"
# Should show gfx950 (MI355X) or gfx1201 (R9700)
```

### Verify ROCm Triton Installation

```python
import triton
print(triton.__version__)
import torch
print(torch.version.hip)  # Should show ROCm version
print(torch.cuda.get_device_properties(0))
```

### Profile Kernel

```bash
# Basic profiling
rocprof --stats python your_kernel.py

# Detailed kernel metrics
rocprofv3 -i metrics.txt python your_kernel.py
```

### Test Kernel Correctness

```python
# Compare with PyTorch reference
ref_output = reference_model(inputs)
custom_output = custom_model(inputs)

torch.testing.assert_close(
    custom_output, ref_output,
    rtol=1e-2, atol=1e-3  # FP16 tolerance
)
```
