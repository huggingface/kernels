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

The decorator does not change the behavior of the class -- it annotates
the class with the given name (here `SiluAndMul`). The `kernelize` function
described below uses this name to look up kernels for the layer.

### External layers

An existing layer that does not (yet) have the `use_kernel_forward_from_hub`
decorator can be made extensible using the `replace_kernel_forward_from_hub`
function:

```python
from somelibrary import SiluAndMul

replace_kernel_forward_from_hub(SiluAndMul, "SiluAndMul")
```

**Warning:** we strongly recommend using layers with a decorator, since
it signifies that the maintainer intends to keep the `forward` signature
compatible with layers from the hub.

## Kernelizing a model

A model will not use Hub kernels by default, even if it contains extensible
layers. To enable the use of Hub kernels in the model, it needs to be
'kernelized' using the `kernelize` function. This function traverses the
model graph and replaces the `forward` methods of extensible layers for which
Hub kernels are registered. Kernelize can be used as follows:

```python
model = MyModel(...)
model = kernelize(model)
```

**Note:** the `kernelize` function modifies the model in-place, the model
itself is returned as a convenience.

### Kernel device

Kernels can be registered per device type. For instance, separate `cuda` and
`metal` kernels could be registered for the name `SiluAndMul`. By default,
`kernelize` will try to infer the device type from the model's parameters.
You can pass the device type to `kernelize` if the device type cannot be
inferred (e.g. because the model has no parameters):

```python
model = MyModel(...)
model = kernelize(model, device="cuda")
```

### Backprop

Not all Hub kernels support backprop. If a model is going to be used with
backprop for training, then we want to avoid that `kernelize` uses such
kernels. By default, `kernelize` will exclude inference-only kernels when
the kernelized model is set to training (`Model.train()`). You can override
this behavior using the `needs_backwards`:

```python
model = MyModel(...)
model = kernelize(model, needs_backward=False)
```

### `torch.compile`

Not all Hub kernels support `torch.compile`. If you want to compile a model
after kernelizing it, pass the `needs_torch_compile` argument to ensure that
only kernels that support `torch.compile` will be loaded:

```python
model = MyModel(...)
model = kernelize(model, needs_torch_compile=True)
```

### Fallback forward

The `needs_backwards`/`needs_torch_compile` arguments will fall back to
the layer's original `forward` if the registered kernels do not support
backprop/`torch.compile`. You can let `kernelize` raise an exception
instead by using `use_fallback=False`:

```python
model = MyModel(...)
model = kernelize(model, needs_torch_compile=True, use_fallback=False)
```

This can be useful if you want to guarantee that Hub kernels are used.

## Registering a hub kernel for a layer

`kernelize`` relies on kernel mappings to find Hub kernels for layers.
Kernel mappings map a kernel name such as `SiluAndMul` to a kernel on
the Hub. For example:

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            revision="layers",
        )
    }
}
```

You can register such a mapping using `register_kernel_mapping`:

```python
register_kernel_mapping(kernel_layer_mapping)
```

This will register the kernel mapping in the current context, which is
normally global. It is recommended to scope the mapping to where it is
used with the `use_kernel_mapping` context manager:

```python
with use_kernel_mapping(kernel_layer_mapping):
    # Use the layer for which the mapping is applied.
    model = kernelize(model)
```

This ensures that the mapping is not active anymore outside the
`with`-scope.
