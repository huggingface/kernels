# Layers

A kernel can provide layers in addition to kernel functions. A layer from
the Hub can replace the `forward` method of an existing layer for a certain
device type. This makes it possible to provide more performant kernels for
existing layers.

See [Kernel requirements](kernel-requirements.md) for more information the
requirements of Hub layers.

## Making a layer extensible with kernels from the hub

### Using a decorator

A layer can be made extensible with the `use_kernel_forward_from_hub`
decorator. For example:

```python
@use_kernel_forward_from_hub("SiluAndMul")
class SiluAndMul(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.shape[-1] // 2
        return F.silu(input[..., :d]) * input[..., d:]
```

The decorator changes the layer, so that other implementations of the `forward`
method can be registered using the name `SiluAndMul`.

### External layers

An existing layer that does not (yet) have the `use_kernel_forward_from_hub`
decorator can be made extensible by by monkeypatching it using the `replace_kernel_forward_from_hub` function.

```python
from somelibrary import SiluAndMul

replace_kernel_forward_from_hub(SiluAndMul, "SiluAndMul")
```

**Warning:** we strongly recommend using layers with a decorator, since
it signifies that the maintainer intends to keep the `forward` signature
compatible with layers from the hub.

## Registering a hub kernel for a layer

Once a layer is made extensible, users can register hub kernels for it
by name using the `register_kernel_mapping` function. For example:

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        Device(type="cuda"): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            revision="layers",
        )
    }
}

register_kernel_mapping(kernel_layer_mapping)
```

This will register the kernel mapping in the current context, which is
normally global. It is recommended to scope the mapping to where it is
used with the `use_kernel_mapping` context manager:

```python
with use_kernel_mapping(kernel_layer_mapping):
    # Use the layer for which the mapping is applied.
    ...
```

This ensures that the mapping is not active anymore outside the
`with`-scope.
