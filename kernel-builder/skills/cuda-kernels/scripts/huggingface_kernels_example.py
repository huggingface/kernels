#!/usr/bin/env python3
"""
Example: Using HuggingFace Kernels library to load and use optimized CUDA kernels.

This script demonstrates how to:
1. Load kernels from the HuggingFace Hub using get_kernel()
2. Check kernel availability with has_kernel()
3. Integrate Hub kernels with transformers/diffusers models

Requirements:
    pip install kernels torch numpy

Usage:
    python huggingface_kernels_example.py
"""

import time
from typing import Optional

import torch
import torch.nn as nn


def check_environment():
    """Print environment information for debugging."""
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU capability: {torch.cuda.get_device_capability()}")
    print()


# =============================================================================
# Part 1: Basic Kernel Loading
# =============================================================================

def demo_basic_kernel_loading():
    """Demonstrate basic kernel loading from Hub."""
    print("=" * 60)
    print("Part 1: Basic Kernel Loading")
    print("=" * 60)

    try:
        from kernels import get_kernel, has_kernel

        # Check if kernel is available for our environment
        repo_id = "kernels-community/activation"

        print(f"\n1. Checking kernel availability: {repo_id}")
        if has_kernel(repo_id):
            print("   Kernel is available for this environment")

            # Load the kernel
            print(f"\n2. Loading kernel from Hub...")
            activation = get_kernel(repo_id, version=1)

            # Inspect available functions
            print(f"\n3. Available functions:")
            functions = [f for f in dir(activation) if not f.startswith('_')]
            for func in functions[:10]:  # Show first 10
                print(f"   - {func}")
            if len(functions) > 10:
                print(f"   ... and {len(functions) - 10} more")

            # Test the kernel
            print(f"\n4. Testing gelu_fast kernel...")
            x = torch.randn((4, 4), dtype=torch.float16, device="cuda")
            y = torch.empty_like(x)

            # Run kernel
            activation.gelu_fast(y, x)
            print(f"   Input shape: {x.shape}")
            print(f"   Output shape: {y.shape}")
            print(f"   Success!")

            return activation
        else:
            print("   No compatible build available for this environment")
            return None

    except ImportError:
        print("\n   kernels library not installed. Install with: pip install kernels")
        return None
    except Exception as e:
        print(f"\n   Error: {e}")
        return None


# =============================================================================
# Part 2: Benchmark Hub Kernel vs PyTorch
# =============================================================================

def demo_benchmark(activation_kernel):
    """Benchmark Hub kernel against PyTorch implementation."""
    print("\n" + "=" * 60)
    print("Part 2: Benchmark Hub Kernel vs PyTorch")
    print("=" * 60)

    if activation_kernel is None:
        print("   Skipping (kernel not loaded)")
        return

    # Test sizes
    sizes = [(1024, 2048), (4096, 4096), (8192, 8192)]

    for size in sizes:
        x = torch.randn(size, dtype=torch.float16, device="cuda")
        y_hub = torch.empty_like(x)
        y_torch = torch.empty_like(x)

        # Warmup
        for _ in range(5):
            activation_kernel.gelu_fast(y_hub, x)
            y_torch = torch.nn.functional.gelu(x)
        torch.cuda.synchronize()

        # Benchmark Hub kernel
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            activation_kernel.gelu_fast(y_hub, x)
        torch.cuda.synchronize()
        hub_time = (time.perf_counter() - start) / iterations * 1000

        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(iterations):
            y_torch = torch.nn.functional.gelu(x)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / iterations * 1000

        speedup = torch_time / hub_time
        print(f"\n   Shape {size}:")
        print(f"   Hub kernel: {hub_time:.4f} ms")
        print(f"   PyTorch:    {torch_time:.4f} ms")
        print(f"   Speedup:    {speedup:.2f}x")


# =============================================================================
# Part 3: Integration with Models
# =============================================================================

def demo_model_integration():
    """Demonstrate integrating Hub kernels with models."""
    print("\n" + "=" * 60)
    print("Part 3: Integration with Models")
    print("=" * 60)

    try:
        from kernels import get_kernel, has_kernel

        # Check for layer norm kernel
        repo_id = "kernels-community/triton-layer-norm"

        if not has_kernel(repo_id):
            print(f"   {repo_id} not available, skipping")
            return

        print(f"\n1. Loading {repo_id}...")
        layer_norm = get_kernel(repo_id)

        # Check available functions
        print(f"\n2. Available functions:")
        functions = [f for f in dir(layer_norm) if not f.startswith('_')]
        for func in functions:
            print(f"   - {func}")

        # Create a simple model with RMSNorm
        class SimpleModel(nn.Module):
            def __init__(self, hidden_size=2048):
                super().__init__()
                self.norm = nn.RMSNorm(hidden_size)
                self.linear = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                x = self.norm(x)
                x = self.linear(x)
                return x

        print(f"\n3. Creating model and patching RMSNorm...")
        model = SimpleModel().cuda().to(torch.bfloat16)

        # Patch RMSNorm to use Hub kernel
        def patch_rmsnorm(model, kernel):
            for name, module in model.named_modules():
                if isinstance(module, nn.RMSNorm):
                    eps = module.eps

                    def make_forward(mod, epsilon):
                        def forward(x):
                            # Try different function names based on kernel API
                            if hasattr(kernel, 'rms_norm'):
                                return kernel.rms_norm(x, mod.weight, eps=epsilon)
                            elif hasattr(kernel, 'rmsnorm'):
                                return kernel.rmsnorm(x, mod.weight, eps=epsilon)
                            else:
                                # Fallback to original
                                return mod._original_forward(x)
                        return forward

                    module._original_forward = module.forward
                    module.forward = make_forward(module, eps)
                    print(f"   Patched: {name}")

        patch_rmsnorm(model, layer_norm)

        # Test forward pass
        print(f"\n4. Testing forward pass...")
        x = torch.randn(2, 1024, 2048, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            y = model(x)
        print(f"   Input: {x.shape}")
        print(f"   Output: {y.shape}")
        print(f"   Success!")

    except ImportError:
        print("   kernels library not installed")
    except Exception as e:
        print(f"   Error: {e}")


# =============================================================================
# Part 4: Using Local Kernels
# =============================================================================

def demo_local_kernel():
    """Demonstrate loading kernels from local path."""
    print("\n" + "=" * 60)
    print("Part 4: Using Local Kernels")
    print("=" * 60)

    try:
        from kernels import get_local_kernel

        # Example: Load from local path (adjust path as needed)
        local_path = "torch-ext"  # Path to your local kernel

        print(f"\n   To load a local kernel:")
        print(f"   >>> from kernels import get_local_kernel")
        print(f"   >>> kernel = get_local_kernel('{local_path}')")
        print(f"\n   This is useful for development before publishing to Hub.")

    except ImportError:
        print("   kernels library not installed")


# =============================================================================
# Part 5: Publishing Your Own Kernel
# =============================================================================

def demo_publishing_info():
    """Show information about publishing kernels."""
    print("\n" + "=" * 60)
    print("Part 5: Publishing Kernels to Hub")
    print("=" * 60)

    print("""
   To publish your own kernels:

   1. Create project structure:
      my-kernel/
      ├── build.toml
      ├── kernel_src/
      │   └── my_kernel.cu
      └── torch-ext/
          ├── torch_binding.cpp
          ├── torch_binding.h
          └── my_kernel/__init__.py

   2. Configure build.toml:
      [general]
      name = "my_kernel"
      backends = ["cuda"]

      [torch]
      src = ["torch-ext/torch_binding.cpp", "torch-ext/torch_binding.h"]

      [kernel.my_kernel]
      backend = "cuda"
      src = ["kernel_src/my_kernel.cu"]
      depends = ["torch"]
      cuda-capabilities = ["7.5", "8.0", "9.0"]

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
    print("HuggingFace Kernels Integration Example")
    print("=" * 60)

    # Check environment
    check_environment()

    if not torch.cuda.is_available():
        print("CUDA not available. This example requires a GPU.")
        return

    # Run demos
    activation = demo_basic_kernel_loading()
    demo_benchmark(activation)
    demo_model_integration()
    demo_local_kernel()
    demo_publishing_info()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
