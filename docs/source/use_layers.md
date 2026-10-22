[[open-in-colab]]

# Kernelize your model with ease

| Authors |
| :-- |
| [Aritra Roy Gosthipaty](https://huggingface.co/ariG23498) |
| [Sayak Paul](https://huggingface.co/sayakpaul) |
| [Daniël de Kok](https://huggingface.co/danieldk) |  

A compute kernel is a function than can be run on an accelerator (GPUs or TPUs). A kernel can be launched on many compute units (Streaming Multiprocessors in case of NVIDIA GPUs) in parallel which is how it achieves arithmetic density.

There are two main branches while working with kernels:

1. Building kernels
2. Using built kernels

While building kernels is an art form, in this tutorial we will showcase how we can leverage prebuilt kernels in our models. We use [`kernels` library](https://huggingface.co/docs/kernels/index), inspect kernels (from Hugging Face Hub), and use them inside a PyTorch model.

> 🙏 This is an interactive notebook, so feel free to make changes and experiment quickly.

## Installs and Imports

```python
!pip install kernels -qq
```

```python
import sys
import inspect
import logging

logging.basicConfig(level=logging.INFO, force=True)  # important to see the kernels in action

import torch
from torch import nn
from torch.nn import functional as F

from kernels import get_kernel
from kernels import CUDAProperties
from kernels import LayerRepository, Device
from kernels import kernelize, Mode
from kernels import use_kernel_mapping
from kernels import use_kernelized_func
from kernels import use_kernel_forward_from_hub
from kernels import replace_kernel_forward_from_hub

import triton
import triton.language as tl
```

## Download a kernel

If you are looking for a model or a dataset, you go to [Hugging Face Hub](https://huggingface.co/) and select from the wide variety of options. What do you do when you are in search for a compute kernel?

With the introduction of [`kernels`](https://huggingface.co/) you can go to the Hub, click on the [kernels tab](https://huggingface.co/kernels), and choose from the wide variety of options 🤗. Compute kernels are now a seperate repository type on the Hub.

This means, we can:

1. Go the Hub
2. Select the kernel repository
3. Download the kernel and use it

Let's get started!

```python
activation_kernel_id = "kernels-community/activation"
activation_kernel_module = get_kernel(activation_kernel_id, version=1)
```

### Inspect the kernel module

A kernel module usually holds kernel functions and kernel layers. Functions (as one would know) are input and output mapping. The idea behind a kernel function is that they are optimized for the hardware that you are using. A kernel layer on the other hand can be thought of as an optimized PyTorch module.

Let's first inspect what the `kernels-community/activation` module holds for us.

```python
functions_available = inspect.getmembers(activation_kernel_module, inspect.isfunction)
layers_available = inspect.getmembers(activation_kernel_module.layers, inspect.isclass)

print("Functions:")
for function in functions_available:
    print(f" {function[0]}")

print("\nLayers:")
for layer in layers_available:
    print(f" {layer[0]}")
```

## How to use a kernel function?

While this cookbook (or tutorial, what ever you want to call it) is intended on showing how one would kernelize their model using a kernel layer, it is quite important to understand how the kernel functions work.

The idea is to create the input, get the kernel function ready to be used, and then create an empty tensor that will hold the output.

```python
input = torch.randn(100, 100, device="cuda", dtype=torch.float16)
kernel_output = torch.empty_like(input)

torch_output = F.gelu(input)
activation_kernel_module.gelu(kernel_output, input)

print(f"Torch matches Kernel Functions: {torch.allclose(torch_output, kernel_output)}")
```

## Why do we need Kernel Layers?

With kernel functions one would need to:

1. highlight the layer
2. replace the layer's `forward` with the kernel function

We wanted to make this more elegant. With a kernel layer, you can just follow:

1. annotate
2. register
3. `kernelize()`

to achieve the `forward` swap withouth having to hardcode anything. This enhances readability of code, make the intent more visible and also introduces composability.

To see this in action, let's first build a simple model and iterate on it.

```python
class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


model = MLP().to("cuda").eval()
x = torch.randn(64, 1024, device="cuda")

with torch.no_grad():
    ref_out = model(x)

print(ref_out.shape)
```

### Build a triton kernel manually and patch it

Here we build a very simple `gelu` kernel using [`triton`](https://triton-lang.org/main/index.html). Readers don't need to follow this code with great detail, as this is just for demonstration. It is not tuned for the GPUs being used.

```python
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    xf = x.to(tl.float32)
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    tl.store(out_ptr + offsets, y.to(x.dtype), mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


# sanity check against PyTorch
t = torch.randn(10_000, device="cuda")
print("Triton matches F.gelu:", torch.allclose(triton_gelu(t), F.gelu(t), atol=1e-5))
```

To use this triton kernel inside our model, we would have to patch the `forward` of the `self.act` layer with the triton kernel. This is what we do next.

```python
temp_model = MLP().to("cuda").eval()
temp_model.load_state_dict(model.state_dict())

temp_model.act.forward = triton_gelu  # patch activation forward with the kernel

with torch.no_grad():
    triton_out = temp_model(x)
print("Model output matches original:", torch.allclose(triton_out, ref_out, atol=1e-5))
```

Now let's see why this is a really clumsy way of doing things:

1. the lack of composability, everything here is hard-coded and has no leeway to compose models to our liking
2. manually monkey pathcing always leads to convoluted code bases
3. no tooling present around patching methods with kernels

To mitigate these issues and more, we introduce kernel layers.

### Build a model with kernel layer

Here we take you through the process of using kernel layers inside a model. This is a really helpful way to use kernels if you had the freedom to build the model from scratch.

```python
# 1. Annotate the layer
@use_kernel_forward_from_hub("Gelu")
class Gelu(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


# 2. Build the registry mapping
kernel_layer_mapping = {
    "Gelu": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
    }
}


# 3. Build the model (from source)
class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = Gelu()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# 4. Kernelize the model
with use_kernel_mapping(kernel_layer_mapping):
    kernelized_model = kernelize(MLP().to("cuda"), device="cuda", mode=Mode.INFERENCE)
    kernelized_model.load_state_dict(model.state_dict())  # load the previous weights for equivalence testing
    with torch.no_grad():
        hub_out = kernelized_model(x)

print("Hub kernel output matches original:", torch.allclose(hub_out, ref_out, atol=1e-5))
```

We went through a bunch of steps all of which achieved the patching (as shown in the triton example above) in a more elegant way.

1. We annotate the layer that we want to replace with a kernel layer form the Hub.
2. We then create a mapping for the layer. Here we map the layer to the kernel id and the actual layer we want to use.
3. After that we build the model using our layer.
4. Finally we `kernelize()` the model: This takes care of traversing through the model and replacing the layer's `forward` with our kernel layer from the Hub.

If you remember the annotate, map, and kernelize formula you are good to go!

This is quite intuitive in our opinon (but we are biased).

### Replace a layer definition

There is another way to kernelize a model using a kernel layer. If you know the definition of the layer you can use our handy `replace_kernel_forwared_from_hub` API. This annotates the layer definition and `kernelize` will later replace it with the kernel layer.

```python
# the model in question
print(model)
```

```python
# replace nn.GELU with the kernel Gelu layer
replace_kernel_forward_from_hub(nn.GELU, "Gelu")

# register the mapping
kernel_layer_mapping = {
    "Gelu": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
    }
}

# kernelize the model
with use_kernel_mapping(kernel_layer_mapping):
    kernelized_model = kernelize(model.to("cuda"), device="cuda", mode=Mode.INFERENCE)
    kernelized_model.load_state_dict(model.state_dict())  # load the previous weights for equivalence testing
    with torch.no_grad():
        hub_out = kernelized_model(x)

print("Hub kernel output matches original:", torch.allclose(hub_out, ref_out, atol=1e-5))
```

Here we replace the `nn.GELU` torch module with the `Gelu` kernel layer we have. To do that we use the `replace_kernel_forward_from_hub` method. It is also worth noting that the method is equivalent to annotation, modifying the class.

We strongly recommended the model developers to add `use_kernel_forward_from_hub` annotations and that `replace_kernel_forward_from_hub` is really a last resort. It relies on the upstream layer not changing in an incompatible way, whereas using the decorator implies a contract.

### Using kernel layers with functions

There are times that you have a function that behaves exactly like the layer's `forward`. We provide a handy decorator `use_kernel_forward_from_hub` which can be used to decorate the function in concern.

Something to note here is that decoration of functions are really for special cases. It comes in handy when you functions to kernelize, but they cannot replace the function by a layer for e.g. compatibility reasons, API breakage, etc.

```python
@use_kernel_forward_from_hub("Gelu")
def gelu(input):
    return F.gelu(input)


@use_kernelized_func(gelu)
class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))  # notice the use of the function


kernel_layer_mapping = {
    "Gelu": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
    }
}

with use_kernel_mapping(kernel_layer_mapping):
    kernelized_model = kernelize(MLP().to("cuda"), device="cuda", mode=Mode.INFERENCE)
    kernelized_model.load_state_dict(model.state_dict())  # load the previous weights for equivalence testing
    with torch.no_grad():
        hub_out = kernelized_model(x)

print("Hub kernel output matches original:", torch.allclose(hub_out, ref_out, atol=1e-5))
```

### Kernel Mapping

A kernel builder wants maximum usage of their kernels. They take time developing the kernel for one hardware, and then devotes time for other hardwares too. Who are we to stop them?

A user can map different kernels with different device specifications all at the same time. This means if you are using a model on different hardware, all the kernels that are written for the hardware follow you without having to write a single line of extra code!

```python
kernel_layer_mapping = {
    "Gelu": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
        "rocm": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
    }
}
```

There are kernels for specific to inference, training, and some are also torch compilable. These are called modes. Each mode can have their own kernels. Mode is handled inside `kernelize` but we can also specify it with our kernels mapping.

This becomes more explicit and you do not need to kernelize models differently based on the mode (as you would above). The mapping takes care of which kernel to draw from the Hub and replace with.

```python
kernel_layer_mapping = {
    "Gelu": {
        "cuda": {
            Mode.INFERENCE: LayerRepository(
                repo_id="kernels-community/activation-inference-optimized",
                layer_name="Gelu",
                version=1,
            ),
            Mode.TRAINING
            | Mode.TORCH_COMPILE: LayerRepository(
                repo_id="kernels-community/activation-training-optimized",
                layer_name="Gelu",
                version=1,
            ),
        }
    }
}
```

Another really loved feature is the CUDA Capabilites check. One can specify the CUDA compatibility and map them to the kernels that would work! These kinds of tooling makes like so much easier for the builder and the user.

Everything is written out for you, one would just need to check the mapping and they are good to go.

```python
kernel_layer_mapping = {
    "Gelu": {
        Device(
            type="cuda",
            properties=CUDAProperties(min_capability=75, max_capability=89),
        ): LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="Gelu",
            version=1,
        ),
        Device(
            type="cuda",
            properties=CUDAProperties(min_capability=90, max_capability=sys.maxsize),
        ): LayerRepository(
            repo_id="kernels-community/activation-hopper",
            layer_name="Gelu",
            version=1,
        ),
    }
}
```

A side note, which we thought was really interesting. If you provide an empty mapping, it de-kernelizes your model!

```python
with use_kernel_mapping({}):
    undo_kernelized_model = kernelize(kernelized_model, device="cuda", mode=Mode.INFERENCE)
    with torch.no_grad():
        undo_hub_out = undo_kernelized_model(x)
```

## Are kernels useful?

We know this should be at the very top of the cookbook. But knowing that you have read it through, we assume that you already know what kernels get you. In this section we just showcase a simple case where we generate some tokens with a model with and without kernels.

Here we showcase the time required to generate 256 tokens for `gpt-oss-2b`. We benchmark the model with and without using kernels layers. We also showcase how more batch size affects the benchmark.

| ![gpt-oss](https://huggingface.co/datasets/ariG23498/colab-images/resolve/main/gpt-kernels-plot.png) |
| :--: |
| Benchmark `gpt-oss-20b` with and without kernels |

You can reproduce the results:

1. [Benchmarking script](https://gist.github.com/ariG23498/b2f6c78ee40d5a02119566d58d00fc56)
2. Run it with Hugging Face Jobs: `uvx hf jobs uv run --flavor h200 benchmark_gpt.py -d`
3. See the [Jobs live](https://huggingface.co/jobs/ariG23498/6ac7c9f5fee2c9007016b027)

## Conclusion

We hope this interactive cookbook gets you up to speed with how one would use kernels which are built and uploaded to the Hub. We hope to seeing more of you using kernels for your models and also your libraries.

Kernels are really powerful and we want you to use them leaving all the tooling issues for us to tackle! Keep the hardware accelerators warm 🔥.

Special thanks to [Noe](https://huggingface.co/NoeFlandre) for reporting a bug (that is now fixed 😅)