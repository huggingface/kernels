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

The `kernels` Python package is used to load kernels from the Hub.

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
activation = get_kernel("kernels-community/activation", version=1)

# Random tensor
x = torch.randn((10, 10), dtype=torch.float16, device="cuda")

# Run the kernel
y = torch.empty_like(x)
activation.gelu_fast(y, x)

print(y)
```

You can [search for kernels](https://huggingface.co/models?other=kernels) on
the Hub.

## 📚 Documentation

Read the [documentation of kernels](https://huggingface.co/docs/kernels/).

## Test coverage

To reproduce the coverage number reported on PRs locally:

```bash
uv run pytest --cov=kernels --cov-report=term-missing tests
```

CI measures coverage on a single canonical matrix cell (Python 3.10 / Torch 2.12.0) and posts a sticky comment on the PR; the threshold is 80% (warn-only — the check stays green either way).
