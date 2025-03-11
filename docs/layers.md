# Layers

A kernel can provide layers in addition to kernel functions. For Torch
kernels, layers are subclasses of [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).

## Getting a layer

Layers are exposed through the `layers` attribute of a kernel,
so they can be used directly after loading a kernel with `get_kernel`.
For example:

```python
import kernels
import torch

activation = kernels.get_kernel("kernels-community/activation", revision="layers")
layer = activation.layers.SiluAndMul()
out = layer(torch.randn((64, 64), device='cuda:0'))
```

## Using a kernel layer as a replacement for an existing layer

An existing layer in a library can be Kernel Hub-enabled using the
`use_hub_kernel` decorator. This decorator will replace the existing
layer if the kernel layer could be loaded successfully.

For example:

```python
@use_hub_kernel(
    "kernels-community/activation",
    layer_name="SiluAndMul",
    revision="layers",
)
class SiluAndMul(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.shape[-1] // 2
        return F.silu(input[..., :d]) * input[..., d:]
```
