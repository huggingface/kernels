# Installation

Install the `kernels` package with `pip` (requires `torch>=2.5` and CUDA):

```bash
pip install kernels
```

or with `uv`

```bash
uv pip install kernels
```

or if you want the latest version from the `main` branch:

```bash
pip install "kernels[benchmark] @ git+https://github.com/huggingface/kernels#subdirectory=kernels"
```

> [!IMPORTANT] 
> We strongly recommend not using a free-threaded Python build yet.
These builds are not only experimental, but do not support the stable ABI
on Python versions before 3.15. Kernels are compiled with the stable ABI
to support a wide range of Python versions.
