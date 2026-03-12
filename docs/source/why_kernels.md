# Why kernels?

Our goal with the `kernels` package is manifold:

* Establish a standard way of structuring and building kernels. The
`builder` component takes kernel source in a pre-defined layout, with
a declarative build configuration, and produces compiled kernels for
a wide matrix of compute backends (e.g., CUDA, ROCM, and XPU), operating
systems, and architectures. The builder also enforces the reproducibility of
the build environment and build steps.
* Provide a standard way of distributing and loading kernels. Kernels
are distributed through the Hugging Face Hub and can be loaded through
the kernels Python package. `kernels` fetches the right kernel build for
the system that it runs, avoiding long local, sometimes hours-long,
builds. It also supports loading of multiple versions of the same
kernel, effectively eliminating "dependency hell".
* Provide kernel builds across all supported PyTorch versions,
accelerators, and capabilities. This is particularly important
because local kernel builds can become unusable once the base
machine learning framework (e.g., PyTorch) is updated to a recent version.

Additionally, there are several advantages to hosting the pre-built
kernels on the Hugging Face Hub platform:

* Trends -- kinds of models using a kernel more than others,
[applications](https://huggingface.co/spaces) using a particular kernel, etc.
* Users immediately know if their hardware supports a kernel
without having to run any installation locally.
* General features of the Hub platform:
    * Seamless versioning
    * Fast download and upload powered by [XET](https://huggingface.co/docs/hub/xet/index)
    * Visibility into the download stats
