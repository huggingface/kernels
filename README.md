# hf-kernels

Make sure you have `torch==2.5.1+cu124` installed.

```python
import torch

from hf_kernels import get_kernel

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

## Locking kernel versions

Projects that use `setuptools` can lock the kernel versions that should be
used. First specify the accepted versions in `pyproject.toml` and make
sure that `hf-kernels` is a build dependency:

```toml
[build-system]
requires = ["hf-kernels", "setuptools"]
build-backend = "setuptools.build_meta"

[tool.kernels.dependencies]
"kernels-community/activation" = ">=0.0.1"
```

Then run `hf-kernel lock .` in the project directory. This generates a `kernels.lock` file with
the locked revisions. The locked revision will be used when loading a kernel with
`get_locked_kernel`:

```python
from hf_kernels import get_locked_kernel

activation = get_locked_kernel("kernels-community/activation")
```

**Note:** the lock file is included in the package metadata, so it will only be visible
to `hf-kernels` after doing an (editable or regular) installation of your project.

## Pre-downloading locked kernels

Locked kernels can be pre-downloaded by running `hf-kernel download .` in your
project directory. This will download the kernels to your local Hugging Face
Hub cache.

The pre-downloaded kernels are used by the `get_locked_kernel` function.
`get_locked_kernel` will download a kernel when it is not pre-downloaded. If you
want kernel loading to error when a kernel is not pre-downloaded, you can use
the `load_kernel` function instead:

````python
```python
from hf_kernels import load_kernel

activation = load_kernel("kernels-community/activation")
````
