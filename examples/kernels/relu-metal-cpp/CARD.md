---
library_name: kernels
license: apache-2.0
---

This is the repository card of kernels-test/relu-metal-cpp that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("kernels-test/relu-metal-cpp", version=1)
relu = kernel_module.relu

relu(...)
```

## Available functions
- `relu`

## Benchmarks

No benchmark available yet.
