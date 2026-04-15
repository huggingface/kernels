#!/usr/bin/env python3
"""
Example: Using HuggingFace Kernels library to load and use optimized kernels on ROCm.

This script demonstrates how to:
1. Load kernels from the HuggingFace Hub using get_kernel()
2. Check kernel availability with has_kernel()
3. Integrate Hub kernels with transformers/diffusers models
4. Fall back to local Triton kernels when Hub builds are unavailable

Requirements:
    python -m pip install -r skills/rocm-kernels/scripts/requirements.txt

Usage:
    python scripts/huggingface_kernels_example.py
"""

import os
import time
from typing import Optional

os.environ['TRITON_HIP_USE_BLOCK_PINGPONG'] = '1'
os.environ['TRITON_HIP_USE_ASYNC_COPY'] = '1'

import torch
import torch.nn as nn
import triton
import triton.language as tl


# =============================================================================
# Local Triton RMSNorm (fallback when Hub kernel unavailable)
# =============================================================================

EPS_DEFAULT = 1e-6

@triton.jit
def rmsnorm_fwd_kernel(
    x_ptr, weight_ptr, out_ptr,
    stride_x, D, eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / D
    rms_inv = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    out = x * rms_inv * w
    tl.store(out_ptr + row * stride_x + offs, out.to(x.dtype), mask=mask)


def local_triton_rmsnorm(x, weight, eps=EPS_DEFAULT):
    """Local Triton RMSNorm — used as fallback when Hub kernel is unavailable."""
    x_2d = x.contiguous().view(-1, x.shape[-1])
    out = torch.empty_like(x_2d)
    M, D = x_2d.shape
    BLOCK_D = triton.next_power_of_2(D)
    num_warps = 4 if BLOCK_D <= 1024 else (8 if BLOCK_D <= 4096 else 16)
    rmsnorm_fwd_kernel[(M,)](
        x_2d, weight, out, x_2d.stride(0), D, float(eps),
        BLOCK_D=BLOCK_D, num_warps=num_warps, num_stages=2,
    )
    return out.view_as(x)


# =============================================================================
# Part 1: Check Environment
# =============================================================================

def check_environment():
    """Print environment information for debugging."""
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"HIP version: {getattr(torch.version, 'hip', 'N/A')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()


# =============================================================================
# Part 2: Basic Kernel Loading from Hub
# =============================================================================

def demo_basic_kernel_loading():
    """Demonstrate basic kernel loading from Hub."""
    print("=" * 60)
    print("Part 1: Basic Kernel Loading from Hub")
    print("=" * 60)

    try:
        from kernels import get_kernel, has_kernel

        repo_id = "kernels-community/triton-layer-norm"

        print(f"\n1. Checking kernel availability: {repo_id}")
        if has_kernel(repo_id):
            print("   Kernel is available for this ROCm environment")

            print(f"\n2. Loading kernel from Hub...")
            kernel = get_kernel(repo_id)

            print(f"\n3. Available functions:")
            functions = [f for f in dir(kernel) if not f.startswith('_')]
            for func in functions[:10]:
                print(f"   - {func}")

            print(f"\n4. Testing RMSNorm kernel...")
            x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
            w = torch.ones(2048, dtype=torch.bfloat16, device="cuda")

            rms_fn_name = None
            for name in ('rms_norm', 'rms_norm_fn', 'rmsnorm'):
                if hasattr(kernel, name):
                    rms_fn_name = name
                    break

            if rms_fn_name:
                rms_fn = getattr(kernel, rms_fn_name)
                try:
                    out = rms_fn(x, w, eps=1e-6)
                except TypeError:
                    # rms_norm_fn(x, weight, bias, ...) requires bias argument
                    out = rms_fn(x, w, None, eps=1e-6)
                print(f"   Using: kernel.{rms_fn_name}()")
                print(f"   Input: {x.shape}, Output: {out.shape}")
                print(f"   Success!")
            else:
                print(f"   No RMSNorm function found. Available: {functions}")

            return kernel
        else:
            print("   No compatible build for this ROCm environment")
            print("   Will use local Triton kernel as fallback")
            return None

    except ImportError:
        print("\n   kernels library not installed. Install with: pip install kernels")
        return None
    except Exception as e:
        print(f"\n   Error: {e}")
        return None


# =============================================================================
# Part 3: Benchmark Hub Kernel vs Local Triton vs PyTorch
# =============================================================================

def demo_benchmark(hub_kernel):
    """Benchmark Hub kernel vs local Triton vs PyTorch."""
    print("\n" + "=" * 60)
    print("Part 2: Benchmark Hub vs Local Triton vs PyTorch")
    print("=" * 60)

    shapes = [(2, 1024, 2048), (4, 4096, 3072)]
    warmup, iterations = 20, 100

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(shape[-1], dtype=torch.bfloat16, device="cuda")

        def _call_hub(fn, x, w, eps):
            try:
                return fn(x, w, eps=eps)
            except TypeError:
                return fn(x, w, None, eps=eps)

        hub_rms_fn_raw = None
        if hub_kernel:
            for fn_name in ('rms_norm', 'rms_norm_fn', 'rmsnorm'):
                if hasattr(hub_kernel, fn_name):
                    hub_rms_fn_raw = getattr(hub_kernel, fn_name)
                    break

        # Warmup all implementations
        for _ in range(warmup):
            local_triton_rmsnorm(x, w, eps=1e-6)
            variance = x.pow(2).mean(-1, keepdim=True)
            _ = x * torch.rsqrt(variance + 1e-6) * w
            if hub_rms_fn_raw:
                _call_hub(hub_rms_fn_raw, x, w, 1e-6)
        torch.cuda.synchronize()

        # PyTorch baseline
        start = time.perf_counter()
        for _ in range(iterations):
            variance = x.pow(2).mean(-1, keepdim=True)
            _ = x * torch.rsqrt(variance + 1e-6) * w
        torch.cuda.synchronize()
        pt_ms = (time.perf_counter() - start) / iterations * 1000

        # Local Triton
        start = time.perf_counter()
        for _ in range(iterations):
            local_triton_rmsnorm(x, w, eps=1e-6)
        torch.cuda.synchronize()
        local_ms = (time.perf_counter() - start) / iterations * 1000

        print(f"\n   Shape {shape}:")
        print(f"   PyTorch:      {pt_ms:.4f} ms")
        print(f"   Local Triton: {local_ms:.4f} ms (speedup: {pt_ms/local_ms:.2f}x)")

        if hub_rms_fn_raw:
            start = time.perf_counter()
            for _ in range(iterations):
                _call_hub(hub_rms_fn_raw, x, w, 1e-6)
            torch.cuda.synchronize()
            hub_ms = (time.perf_counter() - start) / iterations * 1000
            print(f"   Hub kernel:   {hub_ms:.4f} ms (speedup: {pt_ms/hub_ms:.2f}x)")


# =============================================================================
# Part 4: Model Integration with Fallback
# =============================================================================

def demo_model_integration(hub_kernel):
    """Demonstrate integrating kernels with models, with fallback."""
    print("\n" + "=" * 60)
    print("Part 3: Model Integration with Fallback")
    print("=" * 60)

    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=2048):
            super().__init__()
            self.norm = nn.RMSNorm(hidden_size)
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            return self.linear(self.norm(x))

    model = SimpleModel().cuda().to(torch.bfloat16)

    # Decide which RMSNorm to use
    hub_rms_fn = None
    if hub_kernel:
        for fn_name in ('rms_norm', 'rms_norm_fn', 'rmsnorm'):
            if hasattr(hub_kernel, fn_name):
                hub_rms_fn = getattr(hub_kernel, fn_name)
                break

    if hub_rms_fn:
        def _hub_rmsnorm(x, w, eps):
            try:
                return hub_rms_fn(x, w, eps=eps)
            except TypeError:
                return hub_rms_fn(x, w, None, eps=eps)
        rmsnorm_fn = _hub_rmsnorm
        source = "Hub kernel"
    else:
        rmsnorm_fn = local_triton_rmsnorm
        source = "Local Triton"

    print(f"\n1. Using {source} for RMSNorm")

    # Patch model
    for name, module in model.named_modules():
        if isinstance(module, nn.RMSNorm):
            raw_eps = getattr(module, 'eps', None)
            eps = float(raw_eps) if raw_eps is not None else 1e-6

            def make_forward(mod, epsilon, fn):
                def forward(x):
                    return fn(x, mod.weight, epsilon)
                return forward

            module.forward = make_forward(module, eps, rmsnorm_fn)
            print(f"   Patched: {name} (eps={eps})")

    # Test
    print(f"\n2. Testing forward pass...")
    x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        y = model(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Success!")


# =============================================================================
# Part 5: Publishing Info
# =============================================================================

def demo_publishing_info():
    """Show information about publishing kernels to Hub."""
    print("\n" + "=" * 60)
    print("Part 4: Publishing Triton Kernels to Hub")
    print("=" * 60)

    print("""
   For Triton kernels (best ROCm compatibility):

   1. Create project structure:
      my-triton-kernel/
      ├── build.toml
      ├── kernel_src/
      │   └── rmsnorm.py          # Triton kernel
      └── torch-ext/
          ├── torch_binding.cpp
          └── my_kernels/__init__.py

   2. Configure build.toml with ROCm support:
      [general]
      name = "my_kernels"
      backends = ["cuda", "rocm"]

   3. Build and publish:
      $ pip install kernel-builder
      $ kernel-builder build
      $ huggingface-cli upload my-username/my-kernel ./dist

   See: https://huggingface.co/docs/kernels
   """)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("HuggingFace Kernels Integration Example (ROCm)")
    print("=" * 60)

    check_environment()

    if not torch.cuda.is_available():
        print("GPU not available. This example requires an AMD GPU with ROCm.")
        return

    hub_kernel = demo_basic_kernel_loading()
    demo_benchmark(hub_kernel)
    demo_model_integration(hub_kernel)
    demo_publishing_info()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
