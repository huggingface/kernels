# kernels

<div align="center">
<a href="https://huggingface.co/kernels">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernels-thumbnail-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernels-thumbnail-light.png">
  <img alt="Kernels" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernels-thumbnail-light.png" style="max-width: 100%;">
</picture>
</a>
<p align="center">
    <a href="https://pypi.org/project/kernels"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/kernels"></a>
    <a href="https://github.com/huggingface/kernels/tags"><img alt="GitHub tag" src="https://img.shields.io/github/v/tag/huggingface/kernels"></a>
    <a href="https://github.com/huggingface/kernels/actions/workflows/test_kernels.yaml"><img alt="Test kernels" src="https://img.shields.io/github/actions/workflow/status/huggingface/kernels/test_kernels.yaml?label=test"></a>
  
</p>
</div>
<hr/>

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://huggingface.co/). To support this kind
of dynamic loading, Hub kernels differ from traditional Python kernel
packages in that they are made to be:

- Portable: a kernel can be loaded from paths outside `PYTHONPATH`.
- Unique: multiple versions of the same kernel can be loaded in the
  same Python process.
- Compatible: kernels must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

## Components

- You can load kernels from the Hub using the [`kernels`](kernels/) Python package.
- If you are a kernel author, you can build your kernels with [kernel-builder](builder/).
- Hugging Face maintains a set of kernels in [kernels-community](https://huggingface.co/kernels-community).

## 🚀 Quick Start

Install the `kernels` Python package with `pip` (requires `torch>=2.5` and CUDA):

```bash
pip install kernels
```

Here is how you would use the [activation](https://huggingface.co/kernels-community/activation) kernels from the Hugging Face Hub:

```python
import torch

from kernels import get_kernel

# Download optimized kernels from the Hugging Face hub
activation = get_kernel("kernels-community/activation", version=1)

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

Browse available kernels at [huggingface.co/kernels](https://huggingface.co/kernels).

## 📚 Documentation

Read the [documentation of kernels and kernel-builder](https://huggingface.co/docs/kernels/).
