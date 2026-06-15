# Troubleshooting Guide

Common issues and solutions when working with H100 CUDA kernels for diffusers.

## Build Issues

### 1. Type Conversion Errors with FP16/BF16

**Problem:** PyTorch compiles with `-D__CUDA_NO_HALF_OPERATORS__` which disables implicit type conversions:
```
error: no suitable conversion function from "__half" to "float" exists
```

**Solution:** Add explicit type conversion helper functions in your .cu files:
```cuda
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Type conversion helpers (required for PyTorch compatibility)
__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x, float*) { return x; }
__device__ __forceinline__ __half from_float(float x, __half*) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float(float x, __nv_bfloat16*) { return __float2bfloat16(x); }

// Usage in kernels:
float val = to_float(input[idx]);
output[idx] = from_float(result, (scalar_t*)nullptr);
```

### 2. Missing CUDA Headers in torch_binding.cpp

**Problem:** Undeclared types `__half`, `__nv_bfloat16`

**Solution:** Include required headers (never `<torch/extension.h>` — it pulls in pybind11, which breaks the ABI3 build):
```cpp
#include <torch/torch.h>
#include <torch/library.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
```

### 3. Build Errors Mentioning pybind11 / Py_LIMITED_API / abi3

**Problem:** Errors like `pybind11 does not support the limited API`, undefined `PyCFunction` variants, or `kernel-builder check-abi` failures.

**Cause:** The kernel uses a disallowed binding pattern — `#include <torch/extension.h>` (which transitively includes pybind11), `PYBIND11_MODULE`, or a hand-written `setup.py` with `torch.utils.cpp_extension.CUDAExtension`. kernel-builder compiles against the Python limited API (ABI3); pybind11 and setuptools extensions cannot be used.

**Solution:** Remove all pybind11 usage and any hand-written `setup.py`. Register ops in `torch-ext/torch_binding.cpp` with `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` + `REGISTER_EXTENSION(TORCH_EXTENSION_NAME)`, build with `nix run .#build-and-copy -L`, and call ops in Python via `from ._ops import ops`. See "Hard Constraints" in SKILL.md.

### 4. Python Can't Find the Op (`torch.ops.my_kernel` has no attribute ...)

**Problem:** Calling `torch.ops.my_kernel.fn(...)` raises `AttributeError`.

**Cause:** kernel-builder registers ops under a hash-suffixed namespace (e.g. `_my_kernel_a1b2c3d`), so the hardcoded name never exists.

**Solution:** Always go through the generated module: `from ._ops import ops; ops.fn(...)`.

### 5. check-config Fails: Invalid Kernel Name or Missing license

**Problem:** `kernel-builder check-config` rejects `build.toml` with `Invalid kernel name` or `missing field 'license'`.

**Solution:** In `[general]`, `name` must be dash-separated lowercase (`my-kernel`, never `my_kernel`) and `license` is required (e.g. `license = "Apache-2.0"`). The Python package directory uses the underscored form: `torch-ext/my_kernel/` for `name = "my-kernel"`.

### 6. Build Fails: "Kernel is not in a git repository"

**Problem:** Nix evaluation fails with `error: Kernel is not in a git repository, this will create a non-reproducible build.`

**Solution:** The kernel project directory must be a git repository with the files committed:
```bash
git init && git add -A && git commit -m "initial kernel"
```

## Performance Issues

### 7. Bank Conflicts in Shared Memory

**Problem:** Poor performance due to shared memory bank conflicts.

**Solution:** Add padding for 32-bank conflict avoidance:
```cuda
__shared__ float data[32][33];  // 33 instead of 32
```

### 8. Poor Occupancy

**Problem:** Low SM utilization.

**Solution:** Check register usage:
```bash
nvcc --ptxas-options=-v your_kernel.cu
```

### 9. Memory Coalescing

**Problem:** Poor memory bandwidth utilization.

**Solution:** Ensure 128-byte aligned accesses for optimal bandwidth.

## Integration Issues

### 10. AttributeError: 'NoneType' has no attribute 'contiguous' (RMSNorm weight is None)

**Problem:** Model uses `elementwise_affine=False`, so `module.weight` is `None`:
```
AttributeError: 'NoneType' object has no attribute 'contiguous'
```

**Root Cause:** LTX-Video transformer blocks use `RMSNorm(dim, elementwise_affine=False)` which has no learnable weight parameter.

**Solution:** Check if weight exists before using it:
```python
has_weight = hasattr(module, 'weight') and module.weight is not None
if has_weight:
    output = rmsnorm(x, module.weight, eps=eps)
else:
    # Create weight of ones
    weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    output = rmsnorm(x, weight, eps=eps)
```

### 11. GEGLU Kernel Not Being Used

**Problem:** You patched GEGLU modules but the kernel isn't being called.

**Diagnosis:** Check what activation the model actually uses:
```python
for name, module in model.named_modules():
    if 'GEGLU' in type(module).__name__:
        print(f"Found GEGLU: {name}")
    if 'GELU' in type(module).__name__:
        print(f"Found GELU: {name}")
```

**Solution:** LTX-Video uses `GELU`, not `GEGLU`. Only patch GEGLU for models that actually use it (e.g., SD3, FLUX).

### 12. Kernel Patching Doesn't Persist Through CPU Offloading

**Problem:** After `enable_model_cpu_offload()`, patched modules don't work correctly.

**Solution:** Inject kernels AFTER loading model to CUDA, BEFORE enabling offloading:
```python
pipe = LTXPipeline.from_pretrained(...)
pipe.to("cuda")  # Move to CUDA first
inject_optimized_kernels(pipe)  # Patch modules
pipe.enable_model_cpu_offload()  # Now enable offloading
```

### 13. isinstance() Check Misses Diffusers Modules

**Problem:** `isinstance(module, torch.nn.RMSNorm)` returns `False` for diffusers modules.

**Root Cause:** Diffusers has its own `RMSNorm` class that is NOT a subclass of `torch.nn.RMSNorm`:
```python
from diffusers.models.normalization import RMSNorm
# This is a DIFFERENT class from torch.nn.RMSNorm!
```

**Solution:** Check by class name instead:
```python
# WRONG - misses diffusers RMSNorm
if isinstance(module, torch.nn.RMSNorm):

# CORRECT - catches all RMSNorm variants
if type(module).__name__ == 'RMSNorm':
```

## torch.compile Compatibility

### 14. Custom Kernels Don't Work with torch.compile

**Problem:** When using `--use-optimized-kernels` with `--compile`, you get an error:
```
torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
```

Or:
```
torch._dynamo.exc.TorchRuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor)
```

**Root Cause:** Custom C++/CUDA kernels that access tensor data pointers directly are not compatible with torch.compile's graph tracing. The compiler needs to trace through the function using "fake tensors" that don't have real data.

**Solution Options:**

1. **Use one or the other (recommended for now):**
   ```bash
   # Option A: Custom kernels (6% speedup)
   python generate_video.py --use-optimized-kernels

   # Option B: torch.compile (34% speedup)
   python generate_video.py --no-optimized-kernels --compile
   ```

2. **Register a fake (meta) implementation for the op (advanced):** ops registered via `TORCH_LIBRARY_EXPAND` in C++ are already proper custom ops — do NOT re-wrap them with `@torch.library.custom_op` in Python. Add a fake impl using the generated `_ops.py` helpers:
   ```python
   import torch
   from ._ops import ops, add_op_namespace_prefix

   @torch.library.register_fake(add_op_namespace_prefix("rmsnorm_forward"))
   def _(out, input, weight, eps):
       return None  # No shape/dtype changes, output written to 'out'
   ```

3. **Use `torch.compiler.allow_in_graph` (limited):**
   ```python
   # This only works if the kernel doesn't access tensor data pointers during tracing
   @torch.compiler.allow_in_graph
   def rmsnorm(input, weight, eps=1e-6):
       out = torch.empty_like(input)
       ops.rmsnorm_forward(out, input.contiguous(), weight.contiguous(), eps)
       return out
   ```
   Note: This approach fails for most C++ extensions because they access data pointers.

### 15. Performance Comparison: Custom Kernels vs torch.compile

| Configuration | End-to-End Speedup | Notes |
|:---|:---:|:---|
| Baseline (neither) | 1.00x | Reference |
| Custom kernels only | 1.06x | 6% faster, works without compilation overhead |
| torch.compile only | 1.34x | 34% faster, requires warm-up compilation |
| Both (future) | TBD | Requires custom op registration |

**Recommendation:** For production workloads with many generations, use `--compile`. For debugging or quick iterations, use `--use-optimized-kernels`.

## Debugging Tips

### Profile Your Kernels

```bash
# NVIDIA Nsight Systems (system-wide overview)
nsys profile -o kernel_profile python your_script.py

# NVIDIA Nsight Compute (detailed kernel analysis)
ncu --set full --csv -o metrics.csv python your_script.py
```

### Verify Kernel Injection

```python
# Check if attention processors were replaced
for name, module in pipe.transformer.named_modules():
    if hasattr(module, 'processor'):
        print(f"{name}: {type(module.processor).__name__}")
        break

# Test a forward pass through patched modules
with torch.inference_mode():
    x = torch.randn(1, 100, 2048, device='cuda', dtype=torch.bfloat16)
    for name, module in pipe.transformer.named_modules():
        if type(module).__name__ == 'RMSNorm':
            out = module(x)
            print(f"RMSNorm forward pass: {x.shape} -> {out.shape}")
            break
```

### Check CUDA Architecture

```bash
# Verify H100 is detected
python -c "import torch; print(torch.cuda.get_device_capability())"
# Should print (9, 0) for H100
```

### Verify Kernels Are Built

```bash
# Check for compiled .so files
ls torch-ext/ltx_kernels/_ops*.so

# Try importing
python -c "from ltx_kernels import rmsnorm; print('OK')"
```
