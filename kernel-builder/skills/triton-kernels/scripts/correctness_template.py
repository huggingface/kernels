"""
Correctness test template for Triton kernels.

Usage:
    python correctness_template.py

Tests your kernel against a PyTorch reference across multiple shapes,
dtypes, and edge cases. Always run this before benchmarking.

To adapt:
1. Replace `triton_kernel_wrapper(x)` with your kernel
2. Replace `torch_reference(x)` with the PyTorch equivalent
3. Adjust test shapes and tolerances for your use case
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


# ============================================================
# Your kernel here (same as in benchmark_template.py)
# ============================================================

def triton_kernel_wrapper(x):
    """Replace with your kernel wrapper."""
    raise NotImplementedError("Replace with your kernel")


def torch_reference(x):
    """Replace with PyTorch equivalent."""
    raise NotImplementedError("Replace with torch reference")


# ============================================================
# Test suite
# ============================================================

def test_shape(shape, dtype=torch.float32, atol=1e-3, rtol=1e-3):
    """Test a single shape."""
    torch.manual_seed(0)
    x = torch.randn(shape, device=DEVICE, dtype=dtype)
    y_triton = triton_kernel_wrapper(x)
    y_ref = torch_reference(x)
    torch.testing.assert_close(y_triton, y_ref, atol=atol, rtol=rtol)


def run_tests():
    print("Testing various shapes...")

    # Standard shapes
    test_cases = [
        # (shape, description)
        ((1, 128), "minimal 1-row"),
        ((4, 4), "tiny square"),
        ((32, 32), "small square"),
        ((1024, 1024), "medium square"),
        ((4096, 4096), "large square"),
        # Irregular shapes (verify masking works)
        ((1823, 781), "irregular"),
        ((1, 1), "single element"),
        ((1, 12345), "single long row"),
        ((10000, 1), "many single-element rows"),
        # Powers of 2 (no masking needed)
        ((512, 512), "power of 2"),
        ((2048, 2048), "large power of 2"),
        # Non-power-of-2 that's close to a boundary
        ((256, 1023), "just under power of 2"),
        ((256, 1025), "just over power of 2"),
    ]

    for shape, desc in test_cases:
        try:
            test_shape(shape)
            print(f"  PASSED: {desc} {shape}")
        except AssertionError as e:
            print(f"  FAILED: {desc} {shape}")
            print(f"    {e}")
        except Exception as e:
            print(f"  ERROR:  {desc} {shape}")
            print(f"    {type(e).__name__}: {e}")

    # Dtype tests
    print("\nTesting dtypes...")
    dtype_configs = [
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 1e-2, 1e-2),  # fp16 has lower precision
        (torch.bfloat16, 1e-2, 1e-2),
    ]

    for dtype, atol, rtol in dtype_configs:
        try:
            test_shape((1024, 1024), dtype=dtype, atol=atol, rtol=rtol)
            print(f"  PASSED: {dtype}")
        except AssertionError as e:
            print(f"  FAILED: {dtype}")
            print(f"    {e}")
        except Exception as e:
            print(f"  ERROR:  {dtype}: {type(e).__name__}: {e}")

    # Determinism test
    print("\nTesting determinism...")
    torch.manual_seed(42)
    x = torch.randn(1024, 1024, device=DEVICE)
    y1 = triton_kernel_wrapper(x)
    y2 = triton_kernel_wrapper(x)
    if torch.equal(y1, y2):
        print("  PASSED: deterministic")
    else:
        max_diff = (y1 - y2).abs().max().item()
        print(f"  WARNING: non-deterministic (max diff: {max_diff})")

    print("\nDone.")


if __name__ == "__main__":
    run_tests()
