import torch

from kernels import get_kernel

print("Starting examples/basic.py demo")

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation")

print("Activation kernel fetched")

# Create tensor
x = torch.arange(1, 10, dtype=torch.float16, device="cuda").view(3, 3)
print("Input tensor created")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print("Kernel successfully executed")

# Check results
expected = torch.tensor([
    [0.8408, 1.9551, 2.9961],
    [4.0000, 5.0000, 6.0000],
    [7.0000, 8.0000, 9.0000]
], device='cuda:0', dtype=torch.float16)
assert torch.allclose(y, expected)

print("Calculated values are exact")
