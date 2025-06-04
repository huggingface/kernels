# kernels

<div align="center">
<img src="https://github.com/user-attachments/assets/64a652f3-0cd3-4829-b3c1-df13f7933569" width="450" height="450" alt="kernel-builder logo">
<p align="center">
    <a href="https://pypi.org/project/kernels"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/kernels"></a>
    <a href="https://github.com/huggingface/kernels/tags"><img alt="GitHub tag" src="https://img.shields.io/github/v/tag/huggingface/kernels"></a>
    <a href="https://github.com/huggingface/kernels/actions/workflows/docker-build-push.yaml"><img alt="Test kernels" src="https://img.shields.io/github/actions/workflow/status/huggingface/kernels/test.yml?label=test"></a>
  
</p>
</div>
<hr/>

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

## 🚀 Quick Start

Install the `kernels` package with `pip` (requires `torch>=2.5` and CUDA):

```bash
pip install kernels
```

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

You can [search for kernels](https://huggingface.co/models?other=kernel) on
the Hub.

## 📚 Documentation

- [Using layers](docs/layers.md)
- [Locking kernel versions](docs/locking.md)
- [Environment variables](docs/env.md)
- [Using kernels in a Docker container](docs/docker.md)
- [Kernel requirements](docs/kernel-requirements.md)
- [Frequently Asked Questions](docs/faq.md)
- [Writing kernels](https://github.com/huggingface/kernel-builder/blob/main/docs/writing-kernels.md) using [kernel-builder](https://github.com/huggingface/kernel-builder/)
