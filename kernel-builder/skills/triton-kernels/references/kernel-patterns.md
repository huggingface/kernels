# Kernel Patterns

Additional Triton kernel patterns beyond what's covered in SKILL.md.

## Dropout (In-SRAM Mask Generation)

The key insight: generate the random mask in SRAM and consume it immediately,
never writing it to DRAM. This reduces memory traffic from ~6N (PyTorch) to 2N.

```python
@triton.jit
def dropout_kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)  # deterministic hash: (seed, offset) -> float in [0,1)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_dropout(x, p, seed):
    assert x.is_contiguous()
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    dropout_kernel[grid](x, output, n, p, seed, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

Key points:
- `tl.rand(seed, offsets)` is a deterministic hash. Same seed + same offset = same
  random value. This gives free reproducibility without storing the mask.
- Inverted dropout: scale by `1/(1-p)` during training so no scaling needed at inference.

## Fused Add + RMSNorm (Residual Connection)

Fuse the residual addition and normalization into one kernel to avoid an extra
DRAM read/write of the intermediate sum:

```python
@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr, residual_ptr, weight_ptr, out_ptr,
    stride_row, D, eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    x = tl.load(x_ptr + row * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row * stride_row + offsets, mask=mask, other=0.0).to(tl.float32)

    # Fused: add residual then normalize
    h = x + res
    variance = tl.sum(h * h, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    out = h * rms_inv * w

    tl.store(out_ptr + row * stride_row + offsets, out.to(tl.float16), mask=mask)
```

Without fusion: 3 DRAM passes (read x, read residual, write sum, read sum, write norm).
With fusion: 2 DRAM passes (read x + residual, write norm).

## Element-wise Activation (SiLU / Swish)

Simple pattern: one program per block of elements, flat indexing.

```python
@triton.jit
def silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)
```

## Fused SiLU * Gate (SwiGLU)

Common in LLMs (LLaMA MLP): gate and up projections are multiplied element-wise
after SiLU on the gate. If they're stored contiguously as `[..., 2*hidden]`:

```python
@triton.jit
def swiglu_kernel(x_ptr, out_ptr, stride_in, stride_out, hidden_size, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_H)
    mask = offsets < hidden_size

    gate = tl.load(x_ptr + row * stride_in + offsets, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(x_ptr + row * stride_in + hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)

    result = (gate * tl.sigmoid(gate)) * value
    tl.store(out_ptr + row * stride_out + offsets, result.to(tl.float16), mask=mask)
```

## Group-Major PID Ordering (Matmul Optimization)

Programs assigned to adjacent PIDs often end up on the same SM and share L2
cache / SRAM. By default, a 2D grid assigns PIDs in row-major order — programs
in the same row share A tiles but need different B tiles.

Rearranging into square groups improves data sharing:

```python
# Convert 1D PID to 2D block coordinates with group-major ordering
pid = tl.program_id(0)
num_blocks_n = triton.cdiv(N, BLOCK_N)
num_blocks_m = triton.cdiv(M, BLOCK_M)

group_id = pid // (GROUP_SIZE * num_blocks_n)
first_row = group_id * GROUP_SIZE
group_size_adj = min(num_blocks_m - first_row, GROUP_SIZE)

pid_m = first_row + ((pid % (GROUP_SIZE * num_blocks_n)) % group_size_adj)
pid_n = (pid % (GROUP_SIZE * num_blocks_n)) // group_size_adj
```

Nearby PIDs form vertical strips (square groups) instead of long rows. Programs
in a group share both A and B tile data through the L2 cache.

Typical improvement: 5-15% for large matrices. The hardware L2 cache already does
some of this, so gains vary.

## Pointer Sliding (K-Loop Optimization)

Instead of recomputing pointer offsets each iteration:

```python
# Slower: recompute each iteration
for k in range(0, K, BLOCK_K):
    rk = k + tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak, ...)

# Faster: slide pointers forward
a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn

for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

Saves a few integer multiply-adds per iteration. Marginal but free.
