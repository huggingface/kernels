# kernels

Make sure you have `torch==2.5.1+cu124` installed.

```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation")

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

## Docker Reference

build and run the reference [example/basic.py](example/basic.py) in a Docker container with the following commands:

```bash
docker build --platform linux/amd64 -t kernels-reference -f docker/Dockerfile.reference .
docker run --gpus all -it --rm -e HF_TOKEN=$HF_TOKEN kernels-reference
```
