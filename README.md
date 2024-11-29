# kernels

Make sure you have `torch==2.5.1+cu124` installed.

```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
layer_norm_kernels = get_kernel("kernels-community/layer-norm")

# Initialize torch Module
optimized_layer_norm_layer = layer_norm_kernels.DropoutAddLayerNorm(128).cuda()

# Forward
x = torch.randn(128).cuda()
print(optimized_layer_norm_layer(x))
```
