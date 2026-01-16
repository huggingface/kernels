# Migrating from older versions

## 0.12

### Adopting kernel versions

Before `kernels` 0.12, kernels could be pulled from a repository
without specifying a version. This is deprecated in kernels 0.12
and will become an error in kernels 0.14. Instead, use of a kernel
should always specify a version (except for local kernels).

Kernels only use a major version. The kernel maintainer is responsible
for never breaking a kernel within a major version and should bump up
the major version if the kernel API changes and/or when support for
older Torch versions is removed.

You can find the versions that are supported by a kernel using the
`kernels versions command`. For example:

```bash
$ kernels versions kernels-community/activation
Version 1: torch210-cxx11-cu126-x86_64-linux, torch210-cxx11-cu128-x86_64-linux, torch210-cxx11-cu130-x86_64-linux, torch27-cxx11-cu118-x86_64-linux, torch27-cxx11-cu126-x86_64-linux, torch27-cxx11-cu128-aarch64-linux, torch27-cxx11-cu128-x86_64-linux ✅, torch28-cxx11-cu126-aarch64-linux, torch28-cxx11-cu126-x86_64-linux, torch28-cxx11-cu128-aarch64-linux, torch28-cxx11-cu128-x86_64-linux, torch28-cxx11-cu129-aarch64-linux, torch28-cxx11-cu129-x86_64-linux, torch29-cxx11-cu126-aarch64-linux, torch29-cxx11-cu126-x86_64-linux, torch29-cxx11-cu128-aarch64-linux, torch29-cxx11-cu128-x86_64-linux, torch29-cxx11-cu130-aarch64-linux, torch29-cxx11-cu130-x86_64-linux
```

The command lists all available versions (here only version 1) with
all the variants that are supported. A check mark is printed after
the variant that is compatible with your current environment.

Code that uses a kernel can be updated as follows:

```python
# Old:
activation = get_kernel("kernels-community/activation")

# New:
activation = get_kernel("kernels-community/activation", version=1)

# Old:
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        ),
    }
}

# New:
kernel_layer_mapping = {
    "SiluAndMul": {
        "cuda": LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            version=1,
        ),
    }
}
```
