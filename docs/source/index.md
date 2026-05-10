# Kernels

<div align="center">
<a href="https://huggingface.co/kernels">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernels-thumbnail-light.png" alt="Kernels"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/kernels/kernels-thumbnail-dark.png" alt="Kernels"/>
</a>
</div>

The Kernel Hub allows Python libraries and applications to load compute
kernels directly from the [Hub](https://huggingface.co/). Kernels are a first-class
repository type on the Hub, with dedicated pages that surface supported
hardware and versions. To support dynamic loading, Hub kernels differ from
traditional Python kernel packages in that they are made to be:

- **Portable**: a kernel can be loaded from paths outside `PYTHONPATH`.
- **Unique**: multiple versions of the same kernel can be loaded in the
  same Python process.
- **Compatible**: kernels must support all recent versions of Python and
  the different PyTorch build configurations (various CUDA versions
  and C++ ABIs). Furthermore, older C library versions must be supported.

Browse available kernels at [huggingface.co/kernels](https://huggingface.co/kernels).

Before loading kernels from the Hub in production code, read [Security model](./security.md).

If you're looking for a more involved "Why kernels?" answer, refer to
[this page](./why_kernels.md).