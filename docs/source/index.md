# Kernels

<div align="center">
<img src="https://github.com/user-attachments/assets/64a652f3-0cd3-4829-b3c1-df13f7933569" width="450" height="450" alt="kernel-builder logo">
</div>

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://hf.co/). To support this kind
of dynamic loading, Hub kernels differ from traditional Python kernel
packages in that they are made to be:

- **Portable**: a kernel can be loaded from paths outside `PYTHONPATH`.
- **Unique**: multiple versions of the same kernel can be loaded in the
  same Python process.
- **Compatible**: kernels must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

You can [search for kernels](https://huggingface.co/models?other=kernel) on
the Hub.
