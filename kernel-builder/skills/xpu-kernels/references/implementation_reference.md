# Implementation Reference

Code templates and patterns for Triton kernel development on Intel XPU.

## Template Selection

Start with a template that matches your kernel type. The core patterns are shown below:
- Basic GEMM (with tensor descriptors and tile swizzling)
- GEMM with fused epilogue
- Reduction operations

## Core Implementation Pattern

Generated kernels must be **self-contained and shareable**. Define all helper functions inline within the kernel file.

```python
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# Helper functions (inline definitions for self-contained kernel)
# ============================================================================

# Constants
kAlpha = tl.constexpr(math.sqrt(2.0 / math.pi))  # For GeLU
kInvLn2 = tl.constexpr(1.4426950408889634)       # For exp2-based ops

@triton.jit
def swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M):
    """Tile swizzling for L2 cache locality"""
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n

@triton.autotune(
    configs=[
        # Large tiles for square GEMMs
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_warps=32, num_stages=2
        ),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_warps=16, num_stages=3
        ),
        # Medium tiles
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_warps=8, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4, 'grf_mode': '256'},
            num_warps=16, num_stages=3
        ),
        # Skinny-M configs (for M < 256)
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 2, 'grf_mode': '256'},
            num_warps=8, num_stages=4
        ),
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 2, 'grf_mode': '256'},
            num_warps=4, num_stages=5
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr,
    # Shapes (as constexpr for better codegen)
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Strides
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Meta-parameters (NO defaults!)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Optimized GEMM kernel for Intel XPU using tensor descriptors."""
    # Tile swizzling (1D grid)
    pid = tl.program_id(0)
    pid_m, pid_n = swizzle_tile(pid, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M)

    # Tensor descriptors (preferred on XPU — better codegen than block pointers)
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    # Accumulator (fp32 for numerical stability)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for off_k in range(0, K, BLOCK_K):
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)
        acc += tl.dot(a, b)

    # Store result
    c_desc = tl.make_tensor_descriptor(
        base=c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    c_desc.store([off_m, off_n], acc)
```

## Model Class Wrapper (ai-bench compatible)

The Model class uses standard `nn.Module` patterns. ai-bench creates the model via `__init__()` and syncs weights using `copy_model_weights()`.

```python
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        self.gemm = nn.Linear(input_size, hidden_size)
        self._packed = False

    def _pack_weights(self):
        """Pack weight transpose once on XPU for fast tl.dot access."""
        device = torch.device("xpu")
        w = self.gemm.weight.data.detach()
        b = self.gemm.bias.data.detach()
        self.weight_t = w.to(device, torch.float16).t().contiguous()
        self.bias_xpu = b.to(device, torch.float16).contiguous()
        self._packed = True

    def forward(self, x):
        device = torch.device("xpu")
        x = x.to(device, torch.float16).contiguous()
        if not self._packed:
            self._pack_weights()

        M, K = x.shape
        N = self.weight_t.shape[1]
        output = torch.empty((M, N), device=device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )
        kernel[grid](
            x, self.weight_t, output,
            M, N, K,
            x.stride(0), x.stride(1),
            self.weight_t.stride(0), self.weight_t.stride(1),
            output.stride(0), output.stride(1),
        )
        return output

# ============================================================================
# Benchmark harness interface (must match *_pytorch.py)
# ============================================================================
batch_size = 1024
input_size = 8192
hidden_size = 8192
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
```

## Example: GEMM Transformation

**Input** (`test_kernels/14_Gemm_Divide_Sum_Scaling_pytorch.py`):
```python
x = torch.matmul(x, self.weight.T)  # Gemm
x = x / 2  # Divide
x = torch.sum(x, dim=1, keepdim=True)  # Sum
x = x * self.scaling_factor  # Scaling
```

**Strategy**:
1. Use tensor descriptors for GEMM (preferred on XPU)
2. Fuse divide into GEMM epilogue (light)
3. Keep sum + scaling in separate reduction kernel (avoid serializing over N)

**Output**: `gemm_kernel` (matmul + divide fused) + `row_sum_kernel` (sum + scaling).

See `references/examples/gemm_activation_optimized.py` for a similar pattern.

## File Naming Convention

Spec YAML files live in `modules/ai-bench/problems/specs/KernelBench/level*/`.
Auto-detection strips suffixes (`_triton`, `_optimized`, `_opt`, `_pytorch`) from filename and searches `level1/`, `level2/`, `level3/`. Override with `--spec` if needed.

## Activation Helpers

```python
# exp2-based sigmoid (faster on XPU)
sigmoid(x) = 1 / (1 + exp2(-x * 1.44269504))

# tanh via sigmoid
tanh(x) = 2*sigmoid(2x) - 1
```

Factor into reusable `@triton.jit` helpers defined inline in your kernel file.
