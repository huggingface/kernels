---
library_name: kernels
license: apache-2.0
---

This is the repository card of kernels-test/extra-data that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("kernels-test/extra-data", version=1)
EASTER_EGG = kernel_module.EASTER_EGG

EASTER_EGG(...)
```

## Available functions
- `EASTER_EGG`
- `relu`

## Available layers
- `ReLU`

## Benchmarks

No benchmark available yet.
