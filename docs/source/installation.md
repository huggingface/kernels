# Installation

> [!WARNING]
> `kernels` has not reached `1.0` yet. Until then, minor releases may contain
> breaking changes. If you depend on `kernels` in a library or application, we
> **strongly recommend pinning a version range** rather than an unbounded
> dependency. For example, in `pyproject.toml`:
>
> ```toml
> dependencies = [
>     "kernels>=0.15,<0.16",
> ]
> ```
>
> or equivalently `kernels~=0.15` (compatible release). This protects your
> project from unexpected breakage when a new `kernels` version is released.

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
> On Windows, we recommend using the Linux version of Torch through
> [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install), since
> many more kernels support Linux. If you want to use GPU acceleration,
> check out the [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2)
> and [PyTorch with DirectML on WSL 2](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-wsl)
> guides.

> [!IMPORTANT]
> We strongly recommend not using a free-threaded Python build yet.
> These builds are not only experimental, but do not support the stable ABI
> on Python versions before 3.15. Kernels are compiled with the stable ABI
> to support a wide range of Python versions.
