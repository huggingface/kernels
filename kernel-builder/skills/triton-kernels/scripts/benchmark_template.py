"""
Benchmark template for Triton kernels.

Usage:
    python benchmark_template.py

Generates a performance plot (PNG) and prints a data table comparing
your Triton kernel against the PyTorch baseline.

To adapt for your kernel:
1. Replace `triton_kernel_wrapper(x)` with your kernel call
2. Replace `torch_reference(x)` with the equivalent PyTorch op
3. Adjust x_vals, args, ylabel, and the gbps/tflops calculation
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


# ============================================================
# Your kernel here
# ============================================================

@triton.jit
def my_kernel(x_ptr, output_ptr, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    """Replace with your kernel implementation."""
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

    tl.store(output_ptr + row_start + offsets, result, mask=mask)


def triton_kernel_wrapper(x):
    """Replace with your wrapper function."""
    assert x.is_contiguous()
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else (8 if BLOCK_SIZE <= 4096 else 16)
    my_kernel[(n_rows,)](x, output, n_cols, x.stride(0), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return output


def torch_reference(x):
    """Replace with the PyTorch equivalent."""
    return torch.softmax(x, dim=-1)


# ============================================================
# Correctness check (always run before benchmarking)
# ============================================================

def test_correctness():
    torch.manual_seed(0)
    # Test with irregular shape to verify masking
    x = torch.randn(1823, 781, device=DEVICE, dtype=torch.float32)
    y_triton = triton_kernel_wrapper(x)
    y_ref = torch_reference(x)
    torch.testing.assert_close(y_triton, y_ref, atol=1e-3, rtol=1e-3)
    print("Correctness: PASSED")


# ============================================================
# Benchmark
# ============================================================

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
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch_reference(x))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_kernel_wrapper(x))

    # For memory-bound kernels: GB/s
    # 2 = one read + one write
    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps

    # For compute-bound kernels (matmul), use TFLOPS instead:
    # tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    # return tflops


if __name__ == "__main__":
    test_correctness()
    benchmark.run(save_path='.', print_data=True)
