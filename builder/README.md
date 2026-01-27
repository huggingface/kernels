# kernel-builder

The kernel-builder is a build system for creating Hub-compatible compute kernels. It handles the complexity of building kernels that are:

- Portable: kernels can be loaded from paths outside `PYTHONPATH`.
- Unique: multiple versions of the same kernel can be loaded in the same Python process.
- Compatible: kernels support all recent versions of Python and the different PyTorch build configurations (various CUDA versions and C++ ABIs).

**Note:** Torch 2.10 builds are still based on PyTorch release candidates.
Typically the ABI does not break during release candidates. If it does,
you have to recompile your kernels with the final 2.10.0 release.

[Join us on Discord](https://discord.gg/H6Tkmd88N3) for questions and discussions!

This repo contains a Nix package that can be used to build custom machine learning kernels for PyTorch. The kernels are built using the [PyTorch C++ Frontend](https://pytorch.org/cppdocs/frontend.html) and can be loaded from the Hub with the [kernels](https://github.com/huggingface/kernels)
Python package.

This builder is a core component of the larger kernel build/distribution system.

## 🚀 Quick Start

We recommend using [Nix](https://nixos.org/download.html) to build kernels. To speed up builds, first enable the Hugging Face binary cache:

```bash
# Install cachix and configure the cache
cachix use huggingface

# Or run once without installing cachix
nix run nixpkgs#cachix -- use huggingface
```

Then quick start a build with:

```bash
cd examples/relu
nix run .#build-and-copy \
  --max-jobs 2 \
  --cores 8 \
  -L
```

Where `--max-jobs` specifies the number of build variant that should be built concurrently and `--cores` the number of CPU cores that should be used per build variant.

The compiled kernel will then be available in the local `build/` directory.
We also provide Docker containers for CI builds. For a quick build:

```bash
# Using the prebuilt container
cd examples/relu
docker run --rm \
  --mount type=bind,source=$(pwd),target=/kernelcode \
  -w /kernelcode ghcr.io/huggingface/kernel-builder:main build
```

See [dockerfiles/README.md](./dockerfiles/README.md) for more options, including a user-level container for CI/CD environments.

## 🎯 Hardware Support

| Hardware   | Kernels Support | Kernel-Builder Support | Kernels Validated in CI | Tier |
| ---------- | --------------- | ---------------------- | ----------------------- | ---- |
| CUDA       | ✓               | ✓                      | ✓                       | 1    |
| ROCm       | ✓               | ✓                      | ✗                       | 2    |
| XPU        | ✓               | ✓                      | ✗                       | 2    |
| Metal      | ✓               | ✓                      | ✗                       | 2    |
| Huawei NPU | ✓               | ✗                      | ✗                       | 3    |

# 📚 Documentation

- [Writing Hub kernels](../docs/source/builder/writing-kernels.md)
- [Building kernels with Nix](../docs/source/builder/nix.md)
- Framework-specific notes:
  - [Metal](../docs/source/builder/metal.md)
- [Building kernels with Docker](../docs/source/builder/docker.md) (for systems without Nix)
- [Local kernel development](../docs/source/builder/local-dev.md) (IDE integration)
- [Kernel security](../docs/source/builder/security.md)
- [Why Nix?](../docs/source/builder/why-nix.md)

## Credits

The generated CMake build files are based on the vLLM build infrastructure.
