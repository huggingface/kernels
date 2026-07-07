"""
Example: Fused Softmax kernel in Triton.

Demonstrates:
- Row-wise reduction with masking
- Numerical stability (subtract max before exp)
- fp32 accumulation
- Correctness test + benchmark vs PyTorch

This kernel reads each row from DRAM once, does all computation in SRAM,
and writes back once. PyTorch's softmax does ~8MN memory ops; this does 2MN.
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def softmax_kernel(x_ptr, output_ptr, n_cols, stride_row, BLOCK_SIZE: tl.constexpr):
    # One program per row
    pid = tl.program_id(0)
    row_start = pid * stride_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # Load row into SRAM. Masked positions get -inf (won't affect max or sum).
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float('-inf')).to(tl.float32)

    # Numerically stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    numerator = tl.exp(x)       # exp(-inf) = 0, harmless in sum
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # Write back. Masked positions are not stored.
    tl.store(output_ptr + row_start + offsets, result, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused softmax along the last dimension."""
    assert x.is_contiguous()
    assert x.ndim == 2

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE must cover the full row (reduction dimension)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else (8 if BLOCK_SIZE <= 4096 else 16)

    # One program per row
    softmax_kernel[(n_rows,)](
        x, output, n_cols, x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
    )
    return output


# ============================================================
# Correctness test
# ============================================================

def test_softmax():
    torch.manual_seed(0)
    # Irregular shape to verify masking
    x = torch.randn(1823, 781, device=DEVICE, dtype=torch.float32)
    y_triton = triton_softmax(x)
    y_ref = torch.softmax(x, dim=-1)
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
        plot_name='softmax-performance',
        args={'M': 4096},
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: triton_softmax(x))

    gbps = 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps


if __name__ == "__main__":
    test_softmax()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)
