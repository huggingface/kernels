## Why kernels?

Why do we want yet-another standard? Our goal with the `kernels`
package is two-fold:

* Establish a much needed standardization in the ecosystem of
custom kernels. This is enforced in the build structure, support
matrices, reproducibility in the package. We are trying to do this
through `builder` component.
* Provide an easy way for users to load and use pre-built compute
kernels, which, otherwise can take a very long time to build. For
example, building something like Flash Attention 3 from source can
take a couple of hours to build locally.
* Provide kernel builds across all supported PyTorch versions,
accelerators, and capabilities. This is particularly important
because local kernel builds can become unusable once the base
runtime (e.g., CUDA) is updated to a recent version.

Additionally, there are several advantages to hosting the pre-built
kernels on the Hugging Face Hub platform:

* Trends -- kinds of models using a kernel more than others, for
example.
* Users immediately know if their hardware would support a kernel
without having to run any installation locally.
* General features of the Hub platform:
    * Versioning.
    * Visibility into the download stats.