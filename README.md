# kernels

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://hf.co/). To support this kind
of dynamic loading, Hub kernels differ from traditional Python kernel
packages in that they are made to be:

- Portable: a kernel can be loaded from paths outside `PYTHONPATH`.
- Unique: multiple versions of the same kernel can be loaded in the
  same Python process.
- Compatible: kernels must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

## Usage

Kernels depends on `torch>=2.5` and CUDA for now. 

Here is how you would use the [activation](https://huggingface.co/kernels-community/activation) kernels from the Hugging Face Hub:

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

These kernels can be built from the [kernel-builder library](https://github.com/huggingface/kernel-builder). 

If you're looking to better understand how these kernels are structured, or looking to build your own kernels, 
please take a look at the following guide: 
[writing kernels](https://github.com/huggingface/kernel-builder/blob/main/docs/writing-kernels.md).

## Installation

To install `kernels`, we recommend installing from the pypi package:

```bash
pip install kernels
```

You should then be able to run the script above (also in [examples/basic.py](examples/basic.py)):
```bash
python examples/basic.py
```

## Docker Reference

build and run the reference [examples/basic.py](examples/basic.py) in a Docker container with the following commands:

```bash
docker build --platform linux/amd64 -t kernels-reference -f docker/Dockerfile.reference .
docker run --gpus all -it --rm -e HF_TOKEN=$HF_TOKEN kernels-reference
```

## Locking kernel versions

Projects that use `setuptools` can lock the kernel versions that should be
used. First specify the accepted versions in `pyproject.toml` and make
sure that `kernels` is a build dependency:

```toml
[build-system]
requires = ["kernels", "setuptools"]
build-backend = "setuptools.build_meta"

[tool.kernels.dependencies]
"kernels-community/activation" = ">=0.0.1"
```

Then run `kernel lock .` in the project directory. This generates a `kernels.lock` file with
the locked revisions. The locked revision will be used when loading a kernel with
`get_locked_kernel`:

```python
from kernels import get_locked_kernel

activation = get_locked_kernel("kernels-community/activation")
```

**Note:** the lock file is included in the package metadata, so it will only be visible
to `kernels` after doing an (editable or regular) installation of your project.

## Pre-downloading locked kernels

Locked kernels can be pre-downloaded by running `kernel download .` in your
project directory. This will download the kernels to your local Hugging Face
Hub cache.

The pre-downloaded kernels are used by the `get_locked_kernel` function.
`get_locked_kernel` will download a kernel when it is not pre-downloaded. If you
want kernel loading to error when a kernel is not pre-downloaded, you can use
the `load_kernel` function instead:

```python
from kernels import load_kernel

activation = load_kernel("kernels-community/activation")
```
