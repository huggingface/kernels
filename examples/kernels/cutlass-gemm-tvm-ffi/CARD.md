---
library_name: kernels
license: apache-2.0
---

This is the repository card of kernels-test/cutlass-gemm-tvm-ffi that has been pushed on the Hub. It was built to be used with the [`kernels` library](https://github.com/huggingface/kernels). This card was automatically generated.

## How to use

```python
# make sure `kernels` is installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("kernels-test/cutlass-gemm-tvm-ffi", version=1)
cutlass_gemm = kernel_module.cutlass_gemm

cutlass_gemm(...)
```

## Available functions
- `cutlass_gemm`

## Benchmarks

No benchmark available yet.
