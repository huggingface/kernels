# Benchmarking Guide

## Two Types of Kernels, Two Metrics

### Memory-bound kernels (softmax, rmsnorm, activations, dropout)

These are bottlenecked by how fast you can read/write DRAM, not by compute.
Measure in **GB/s** (gigabytes per second).

```python
bytes_moved = bytes_read + bytes_written
gbps = bytes_moved * 1e-9 / (ms * 1e-3)
```

For most element-wise and reduction ops, `bytes_moved = 2 * numel * element_size`
(one read + one write).

### Compute-bound kernels (matmul, attention)

These are bottlenecked by arithmetic throughput. Measure in **TFLOPS**.

```python
# Matmul: 2*M*N*K flops (one multiply + one add per output element per K step)
tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
```

## How to Know Which Type You Have

Use the roofline model. Compute the **arithmetic intensity** (flops per byte of
memory traffic):

```
AI = flops / bytes_moved
```

Compare to the machine's compute-to-bandwidth ratio:

```
machine_balance = peak_tflops / peak_bandwidth_TB_s
```

- If AI < machine_balance: memory-bound (measure GB/s)
- If AI > machine_balance: compute-bound (measure TFLOPS)

Examples:
- Softmax: AI ≈ 5 flops/byte → memory-bound on all GPUs
- Matmul (large): AI ≈ M*N*K / (M*K + K*N + M*N) → compute-bound for large dims
- Matmul (small, e.g. 64x64): often memory-bound due to launch overhead

## GPU Bandwidth References

| GPU | HBM Bandwidth | Peak FP32 TFLOPS |
| --- | --- | --- |
| V100 | 900 GB/s | 15.7 |
| A100 | 2.0 TB/s | 19.5 |
| H100 | 3.35 TB/s | 67 |
| MI355X | 8 TB/s | — |

A kernel achieving >70% of theoretical bandwidth is considered well-optimized
for a memory-bound workload.

## triton.testing.do_bench

The standard way to time Triton kernels:

```python
ms = triton.testing.do_bench(lambda: my_kernel(x))
```

This handles:
- GPU warmup (discards first few runs)
- Synchronization (waits for GPU to finish before timing)
- Multiple iterations (averages over many runs)
- Returns milliseconds

Do NOT use `time.time()` or `torch.cuda.Event` manually unless you have a good
reason. `do_bench` handles the subtleties correctly.

## Benchmark Harness

`triton.testing.perf_report` + `Benchmark` gives you plots and CSV output:

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],                          # x-axis parameter
        x_vals=[128 * i for i in range(2, 100)],  # x-axis values
        line_arg='provider',                    # what varies across lines
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='my-kernel-performance',
        args={'M': 4096},                       # fixed args
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

This generates:
- A PNG plot saved to `save_path`
- A CSV with raw data
- Optionally prints a table to stdout

## When to Expect Gains Over PyTorch

Triton kernels typically beat PyTorch when:

1. **Fusion opportunity**: Multiple PyTorch ops that each round-trip through DRAM
   can be fused into one kernel. Example: softmax (5 ops → 1 kernel, ~4x fewer
   memory ops).

2. **Large N**: At small N, kernel launch overhead dominates. Triton kernels have
   slightly higher launch overhead than PyTorch's pre-compiled CUDA kernels. The
   crossover is typically around N=10K-100K elements.

3. **Non-standard shapes**: PyTorch's kernels are tuned for common shapes. For
   unusual dimensions, a Triton kernel with proper masking can outperform.

Triton kernels typically DON'T beat PyTorch when:

1. **The op already has a highly optimized CUDA implementation**: cuBLAS matmul,
   cuDNN convolution, Flash Attention. These have years of hand-tuning.

2. **Small inputs**: Launch overhead dominates.

3. **Single element-wise op**: PyTorch already fuses simple element-wise chains
   via torch.compile.

## Accurate Benchmarking Checklist

1. Use a dedicated CUDA stream:
   ```python
   stream = torch.cuda.Stream()
   torch.cuda.set_stream(stream)
   ```

2. Ensure inputs are on GPU and contiguous before timing.

3. Don't include memory allocation in the timed region.

4. Run correctness test FIRST — a fast but wrong kernel is useless.

5. Test multiple input sizes — performance characteristics change with scale.

6. Report the GPU model, dtype, and input shapes alongside numbers.
