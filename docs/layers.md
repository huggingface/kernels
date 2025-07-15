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
Hub kernels are registered. `kernelize` can be used as follows:

```python
model = MyModel(...)
model = kernelize(model, mode=Mode.INFERENCE)
```

The `kernelize` function modifies the model in-place, the model itself is
returned as a convenience. The `mode` specifies that the model will be used
in inference. Similarly, you can ask `kernelize` to prepare the model for
training:

```python
model = MyModel(...)
model = kernelize(model, mode=Mode.TRAINING)
```

A model that is kernelized for training can also be used for inference, but
not the other way around.

If you want to compile a model with `torch.compile`, this should be indicated
in the mode as well. You can do this by combining `Mode.INFERENCE` or
`Mode.TRAINING` with `Mode.TORCH_COMPILE` using the set union (`|`) operator:

```python
model = MyModel(...)

# Inference
model = kernelize(model, mode=Mode.INFERENCE | Mode.TORCH_COMPILE)

# Training
model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
```

When the `mode` argument is not specified,
`Mode.TRAINING | Mode.TORCH_COMPILE` is used as the default. This mode
aligns most closely with pure PyTorch layers which also support training
and `torch.compile`. However, to select the most performant kernels, it
is often good to make the mode specific as possible.

### Kernel device

Kernels can be registered per device type. For instance, separate `cuda` and
`metal` kernels could be registered for the name `SiluAndMul`. By default,
`kernelize` will try to infer the device type from the model's parameters.
You can pass the device type to `kernelize` if the device type cannot be
inferred (e.g. because the model has no parameters):

```python
model = MyModel(...)
model = kernelize(model, device="cuda", mode=Mode.INFERENCE)
```

### Fallback `forward`

If the `TRAINING` and/or `TORCH_COMPILE` modes are used, but a registered
kernel does not support backward passes or `torch.compile` respectively,
`kernenize` will fall back to the original, non-kernelized, layer. You
can let `kernelize` raise an exception instead by using `use_fallback=False`:

```python
model = MyModel(...)
model = kernelize(model, mode=Mode.INFERENCE | Mode.TORCH_COMPILE, use_fallback=False)
```

This can be useful if you want to guarantee that Hub kernels are used.

## Registering a hub kernel for a layer

`kernelize` relies on kernel mappings to find Hub kernels for layers.
Kernel mappings map a kernel name such as `SiluAndMul` to a kernel on
the Hub. For example:

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
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

### Registering kernels for specific modes

You might want to register two different kernels for a particular layer,
where one kernel is optimized for a specific mode. You can do so by
registering layer repositories for specific modes. For example:

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": {
          Mode.INFERENCE: LayerRepository(
              repo_id="kernels-community/activation-inference-optimized",
              layer_name="SiluAndMul",
          ),
          Mode.TRAINING | Mode.TORCH_COMPILE: LayerRepository(
              repo_id="kernels-community/activation-training-optimized",
              layer_name="SiluAndMul",
          ),
      }
    }
}
```

The `kernelize` function will attempt to use the following registered
kernels for a given mode:

- `INFERENCE`: `INFERENCE` → `INFERENCE | TORCH_COMPILE` → `TRAINING` →
  `TRAINING | TORCH_COMPILE` → `FALLBACK`
- `INFERENCE | TORCH_COMPILE`: `INFERENCE | TORCH_COMPILE` →
  `TRAINING | TORCH_COMPILE` → `FALLBACK`
- `TRAINING`: `TRAINING` → `TRAINING | TORCH_COMPILE` → `FALLBACK`
- `TRAINING | TORCH_COMPILE`: `TRAINING | TORCH_COMPILE` → `FALLBACK`

`Mode.FALLBACK` is a special mode that is used when no other mode matches. It
is also used when a kernel is registered without a mode, as described in the
previous section.

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": {
            Mode.FALLBACK: LayerRepository(
                repo_id="kernels-community/activation",
                layer_name="SiluAndMul",
            ),
            Mode.INFERENCE: LayerRepository(
                repo_id="kernels-community/activation-inference-optimized",
                layer_name="SiluAndMul",
            ),
            Mode.TRAINING: LayerRepository(
                repo_id="kernels-community/activation-training-optimized",
                layer_name="SiluAndMul",
            ),
        }
    }
}
```

In this case, both `Mode.INFERENCE | Mode.TORCH_COMPILE` and
`Mode.TRAINING | Mode.TORCH_COMPILE` will use the `Mode.FALLBACK` kernel,
since the other kernels do not support `torch.compile`.

### Registering kernels for specific CUDA capabilities

Some kernels only work with newer CUDA architectures. For instance, some
kernels require capability 9.0 for the TMA unit on Hopper GPUs. `kernels`
supports registering layers for a range of CUDA capabilities. To do so,
you need to register the layer for a `Device` with type `cuda` and
set the supported range of CUDA capabilities with using `CUDAProperties`:

```python
kernel_layer_mapping = {
    "SiluAndMul": {
        Device(
            type="cuda",
            properties=CUDAProperties(
                min_capability=75, max_capability=89
            ),
        ): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        ),
        Device(
            type="cuda",
            properties=CUDAProperties(
                min_capability=90, max_capability=sys.maxsize
            ),
        ): LayerRepository(
            repo_id="kernels-community/activation-hopper",
            layer_name="SiluAndMul",
        ),
    }
}
```

Capabilities behave as follows:

- The minimum and maximum capabilities are inclusive.
- When a new kernel is registered with the same min/max capabilities as
  an existing kernel, the new kernel will replace the old kernel.
- When there are multiple kernels that support a capability, the kernel
  with the smaller capability interval will be used. E.g. given:

  - `KernelA` with `min_capability=80` and `max_capability=89`;
  - `KernelB` with `min_capability=75` and `max_capability=89`;
  - `kernelize` runs on a system with capability 8.6.

  Then `KernelA` will be used because the interval 80..89 is smaller
  than 75..89. The motivation is that kernels with smaller ranges
  tend to be more optimized for a specific set of GPUs. **This behavior
  might still change in the future.**
