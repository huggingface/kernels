# Kernel requirements

Kernels on the Hub must fulfill the requirements outlined on this page.
You can use [kernel-builder](https://github.com/huggingface/kernel-builder/)
to build conforming kernels.

## Directory layout

A kernel repository on the Hub must contain a `build` directory. This
directory contains build variants of a kernel in the form of directories
following the template
`<framework><version>-cxx<abiver>-<cu><cudaver>-<arch>-<os>`.
For example `build/torch26-cxx98-cu118-x86_64-linux`. The currently
recommended build variants are:

- `torch25-cxx11-cu118-x86_64-linux`
- `torch25-cxx11-cu121-x86_64-linux`
- `torch25-cxx11-cu124-x86_64-linux`
- `torch25-cxx98-cu118-x86_64-linux`
- `torch25-cxx98-cu121-x86_64-linux`
- `torch25-cxx98-cu124-x86_64-linux`
- `torch26-cxx11-cu118-x86_64-linux`
- `torch26-cxx11-cu124-x86_64-linux`
- `torch26-cxx11-cu126-x86_64-linux`
- `torch26-cxx98-cu118-x86_64-linux`
- `torch26-cxx98-cu124-x86_64-linux`
- `torch26-cxx98-cu126-x86_64-linux`

This list will be updated as new PyTorch versions are released. Each
variant directory should contain a single directory with the same name
as the repository (replacing `-` by `_`). For instance, kernels in the
`kernels-community/activation` repository have a directories like
`build/<variant>/activation`. This directory
must be a Python package with an `__init__.py` file.

## Native Python module

Kernels will typically contain a native Python module with precompiled
compute kernels and bindings. This module must fulfill the following
requirements:

- Use [ABI3/Limited API](https://docs.python.org/3/c-api/stable.html#stable-application-binary-interface)
  for compatibility with Python 3.9 and later.
- Compatible with glibc 2.27 or later. This means that no symbols
  from later versions must be used. To archive this, the module should
  be built against this glibc version. **Warning:** libgcc must also be
  built against glibc 2.27 to avoid leaking symbols.
- No dynamic linkage against libstdc++/libc++. Linkage for C++ symbols
  must be static.
- No dynamic library dependencies outside Torch or CUDA libraries
  installed as dependencies of Torch.

(These requirements will be updated as new PyTorch versions are released.)

## Torch extension

Torch native extension functions must be [registered](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
in `torch.ops.<namespace>`. Since we allow loading of multiple versions of
a module in the same Python process, `namespace` must be unique for each
version of a kernel. Failing to do so will create clashes when different
versions of the same kernel are loaded. Two suggested ways of doing this
are:

- Appending a truncated SHA-1 hash of the git commit that the kernel was
  built from to the name of the extension.
- Appending random material to the name of the extension.

**Note:** we recommend against appending a version number or git tag.
Version numbers are typically not bumped on each commit, so users
might use two different commits that happen to have the same version
number. Git tags are not stable, so they do not provide a good way
of guaranteeing uniqueness of the namespace.

## Python requirements

- Python code must be compatible with Python 3.9 and later.
- All Python code imports from the kernel itself must be relative. So,
  for instance if in the example kernel `example`,
  `module_b` needs a function from `module_a`, import as:

  ```python
  from .module_a import foo
  ```

  **Never use:**

  ```python
  # DO NOT DO THIS!

  from example.module_a import foo
  ```

  The latter would import from the module `example` that is in Python's
  global module dict. However, since we allow loading multiple versions
  of a module, we uniquely name the module.

- Only modules from the Python standard library, Torch, or the kernel itself
  can be imported.
