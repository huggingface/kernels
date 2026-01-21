# Using the kernel builder with Docker

<!-- toc -->

- [Using the kernel builder with Docker](#using-the-kernel-builder-with-docker)
  - [Quick Start](#quick-start)
  - [CLI Interface](#cli-interface)
    - [Examples](#examples)
  - [Volume Mounting and Working Directory](#volume-mounting-and-working-directory)
  - [Configuration](#configuration)
    - [1. Environment Variables](#1-environment-variables)
    - [2. Command-line Options](#2-command-line-options)
  - [Development Shell](#development-shell)
    - [Persistent Development Environment](#persistent-development-environment)
  - [Final Output](#final-output)
  - [Reproducible run](#reproducible-run)
    - [Accessing kernel in expected format](#accessing-kernel-in-expected-format)
  - [Building from URL](#building-from-url)
  - [Available Docker Images](#available-docker-images)

**Warning**: we strongly recommend [building kernels with Nix](nix.md).
Using Nix directly makes it easier to cache all dependencies and is more
robust. We provide a Docker image for systems where Nix cannot be
installed.

## Quick Start

We provide a Docker image with which you can build a kernel:

```bash
# navigate to the relu directory
cd examples/relu

# then run the following command to build the kernel
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main
```

This will build the kernel and save the output in the `build` directory in
the relu folder.

## CLI Interface

The kernel builder includes a command-line interface for easier interaction. The following commands are available:

| Command       | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `build`       | Build the kernel extension (default if no command specified) |
| `dev`         | Start a development shell                                    |
| `fetch [URL]` | Clone and build from a Git URL                               |
| `help`        | Show help information                                        |

### Examples

```bash
# Build the example relu kernel from the root of the repository
docker run --rm \
  -v $(pwd):/kernel-builder \
  -w /kernel-builder/examples/relu \
  ghcr.io/huggingface/kernel-builder:main \
  build

# Build from the current directory (assuming it contains a kernel)
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  build

# Start an ephemeral development shell
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  dev

# Build from a Git URL
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  fetch \
  https://huggingface.co/kernels-community/activation.git

# Show help information
docker run --rm ghcr.io/huggingface/kernel-builder:latest help
```

## Volume Mounting and Working Directory

When running the Docker container, you must mount your local volume containing the kernel code to the container and set the working directory appropriately. This is done using the `-v` and `-w` flags in the `docker run` command.

| Flag | Description                                    |
| ---- | ---------------------------------------------- |
| `-v` | Mount a local directory to the container       |
| `-w` | Set the working directory inside the container |

## Configuration

The kernel builder can be configured in two ways:

### 1. Environment Variables

| Variable   | Description                                                         | Default |
| ---------- | ------------------------------------------------------------------- | ------- |
| `MAX_JOBS` | The maximum number of parallel jobs to run during the build process | `4`     |
| `CORES`    | The number of cores to use during the build process                 | `4`     |

```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  -e MAX_JOBS=8 \
  -e CORES=8 \
  ghcr.io/huggingface/kernel-builder:main
```

### 2. Command-line Options

You can also specify these parameters using command-line options:

| Option        | Description                         | Default |
| ------------- | ----------------------------------- | ------- |
| `--jobs, -j`  | Set maximum number of parallel jobs | `4`     |
| `--cores, -c` | Set number of cores per job         | `4`     |

```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  build --jobs 8 --cores 4
```

## Development Shell

For development purposes, you can start an interactive shell with:

```bash
docker run -it \
  --name my-dev-env \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  dev
```

This will drop you into a Nix development shell with all the necessary tools installed.

### Persistent Development Environment

For iterative development, you can create a persistent container to maintain the Nix store cache across sessions:

```bash
# Create a persistent container and start a development shell
docker run -it \
  --name my-persistent-dev-env \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  dev
```

You can restart and attach to this container in subsequent sessions without losing the Nix store cache or the kernel build:

```bash
# Start the container in detached mode
docker start my-persistent-dev-env

# Attach to the container
docker exec -it my-persistent-dev-env sh

# Once inside, start the development shell
/home/nixuser/bin/cli.sh dev
```

This approach preserves the Nix store cache between sessions, making subsequent builds much faster.

## Final Output

The whole goal of building these kernels is to allow researchers, developers, and programmers to use high performance kernels in their PyTorch code. Kernels uploaded to Hugging Face Hub can be loaded using the [kernels](https://github.com/huggingface/kernels/) package.

To load a kernel locally, you should add the kernel build that is compatible with the Torch and CUDA versions in your environment to `PYTHONPATH`. For example:

```bash
# PyTorch 2.9 and CUDA 12.6
export PYTHONPATH="result/torch29-cxx11-cu126-x86_64-linux"
```

The kernel can then be imported as a Python module:

```python
import torch

import relu

x = torch.randn(10, 10)
out = torch.empty_like(x)
relu.relu(x, out)

print(out)
```

## Reproducible run

### Accessing kernel in expected format

Kernels will be available in the [kernel-community](https://huggingface.co/kernels-community) on huggingface.co.

We can reproduce a build of a kernel by cloning the kernel repository and running the build command.

```bash
git clone git@hf.co:kernels-community/activation
cd activation
# then run the build command
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  build
# we should now have the built kernels on our host
ls result
# torch24-cxx11-cu118-x86_64-linux  torch24-cxx98-cu121-x86_64-linux  torch25-cxx11-cu124-x86_64-linux
# torch24-cxx11-cu121-x86_64-linux  torch24-cxx98-cu124-x86_64-linux  torch25-cxx98-cu118-x86_64-linux
# torch24-cxx11-cu124-x86_64-linux  torch25-cxx11-cu118-x86_64-linux  torch25-cxx98-cu121-x86_64-linux
# torch24-cxx98-cu118-x86_64-linux  torch25-cxx11-cu121-x86_64-linux  torch25-cxx98-cu124-x86_64-linux
```

## Building from URL

You can also directly build kernels from a Git repository URL:

```bash
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/huggingface/kernel-builder:main \
  fetch \
  https://huggingface.co/kernels-community/activation.git
```

This will clone the repository into the container, build the kernels, and save the output in the container's `/home/nixuser/kernelcode/build` directory.

## Available Docker Images

The kernel-builder is available in different variants with specific tags:

| Tag          | Description                                                                         |
| ------------ | ----------------------------------------------------------------------------------- |
| `[SHA]`      | Specific commit hash version (example: `ghcr.io/huggingface/kernel-builder:abc123`) |
| `user-[SHA]` | Non root user variant (use when specific permissions are needed)                    |

All images are available from the GitHub Container Registry:

```
ghcr.io/huggingface/kernel-builder
```
