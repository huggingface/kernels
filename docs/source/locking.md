# Locking kernel versions

Projects that use `setuptools` can lock the kernel versions that should be
used. First specify the accepted versions in `pyproject.toml` and make
sure that `kernels` is a build dependency:

```toml
[build-system]
requires = ["kernels", "setuptools"]
build-backend = "setuptools.build_meta"

[tool.kernels.dependencies]
"kernels-community/activation" = ">=0.0.1"
```

Then run `kernels lock .` in the project directory. This generates a `kernels.lock` file with
the locked revisions. The locked revision will be used when loading a kernel with
`get_locked_kernel`:

```python
from kernels import get_locked_kernel

activation = get_locked_kernel("kernels-community/activation")
```

**Note:** the lock file is included in the package metadata, so it will only be visible
to `kernels` after doing an (editable or regular) installation of your project.

## Pre-downloading locked kernels

Locked kernels can be pre-downloaded by running `kernels download .` in your
project directory. This will download the kernels to your local Hugging Face
Hub cache.

The pre-downloaded kernels are used by the `get_locked_kernel` function.
`get_locked_kernel` will download a kernel when it is not pre-downloaded. If you
want kernel loading to error when a kernel is not pre-downloaded, you can use
the `load_kernel` function instead:

```python
from kernels import load_kernel

activation = load_kernel("kernels-community/activation")
```
