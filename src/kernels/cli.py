import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from huggingface_hub import create_branch, create_repo, upload_folder

from kernels.compat import tomllib
from kernels.lockfile import KernelLock, get_kernel_locks
from kernels.utils import install_kernel, install_kernel_all_variants

from .doc import generate_readme_for_kernel

BUILD_VARIANT_REGEX = re.compile(r"^(torch\d+\d+|torch-)")

# Default templates for kernel project initialization
DEFAULT_FLAKE_NIX = """\
{
  description = "Flake for %(kernel_name)s kernel";
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      path = ./.;
      rev = self.shortRev or self.dirtyShortRev or self.lastModifiedDate;
    };
}
"""

DEFAULT_BUILD_TOML = """\
[general]
name = "%(kernel_name)s"
backends = ["cuda", "rocm", "metal", "xpu"]

[torch]
src = [
    "torch-ext/torch_binding.cpp",
    "torch-ext/torch_binding.h",
]

[kernel.activation_cuda]
backend = "cuda"
# cuda-capabilities = ["9.0", "10.0", "12.0"] # if not specified, all capabilities will be used
depends = ["torch"]
src = [
    "%(kernel_name)s_cuda/kernel.cu",
    "%(kernel_name)s_cuda/kernel.h"
]

[kernel.activation_rocm]
backend = "rocm"
# rocm-archs = ["gfx906", "gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx1030", "gfx1100", "gfx1101"] # if not specified, all architectures will be used
depends = ["torch"]
src = [
    "%(kernel_name)s_cuda/kernel.cu",
    "%(kernel_name)s_cuda/kernel.h",
]

[kernel.activation_xpu]
backend = "xpu"
depends = ["torch"]
src = [
    "%(kernel_name)s_xpu/kernel.cpp",
    "%(kernel_name)s_xpu/kernel.hpp",
]

[kernel.activation_metal]
backend = "metal"
depends = ["torch"]
src = [
    "%(kernel_name)s_metal/kernel.mm",
    "%(kernel_name)s_metal/kernel.metal",
]
"""

DEFAULT_GITATTRIBUTES = """\
*.so filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
"""

DEFAULT_README = """\
# %(kernel_name)s

A custom kernel for PyTorch.

## Installation

```bash
pip install kernels
```

## Usage

```python
from kernels import get_kernel

kernel = get_kernel("%(repo_id)s")


## License

Apache-2.0
"""

DEFAULT_INIT_PY = """\
# %(kernel_name)s kernel
# This file exports the kernel's public API

import torch
from ._ops import ops

def exported_kernel_function(x: torch.Tensor) -> torch.Tensor:
    return ops.kernel_function(x)

__all__ = ["exported_kernel_function"]
"""

DEFAULT_TORCH_BINDING_CPP = """\
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("kernel_function(Tensor input) -> (Tensor)");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
  ops.impl("kernel_function", torch::kCUDA, &kernel_function);
#elif defined(METAL_KERNEL)
  ops.impl("kernel_function", torch::kMPS, kernel_function);
#elif defined(XPU_KERNEL)
  ops.impl("kernel_function", torch::kXPU, &kernel_function);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

"""

DEFAULT_TORCH_BINDING_H = """\
#pragma once

#include <torch/torch.h>

torch::Tensor kernel_function(torch::Tensor input);
"""


def main():
    parser = argparse.ArgumentParser(
        prog="kernel", description="Manage compute kernels"
    )
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser("check", help="Check a kernel for compliance")
    check_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    check_parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The kernel revision (branch, tag, or commit SHA, defaults to 'main')",
    )
    check_parser.add_argument("--macos", type=str, help="macOS version", default="15.0")
    check_parser.add_argument(
        "--manylinux", type=str, help="Manylinux version", default="manylinux_2_28"
    )
    check_parser.add_argument(
        "--python-abi", type=str, help="Python ABI version", default="3.9"
    )
    check_parser.set_defaults(
        func=lambda args: check_kernel(
            macos=args.macos,
            manylinux=args.manylinux,
            python_abi=args.python_abi,
            repo_id=args.repo_id,
            revision=args.revision,
        )
    )

    download_parser = subparsers.add_parser("download", help="Download locked kernels")
    download_parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    download_parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Download all build variants of the kernel",
    )
    download_parser.set_defaults(func=download_kernels)

    upload_parser = subparsers.add_parser("upload", help="Upload kernels to the Hub")
    upload_parser.add_argument(
        "kernel_dir",
        type=Path,
        help="Directory of the kernel build",
    )
    upload_parser.add_argument(
        "--repo-id",
        type=str,
        help="Repository ID to use to upload to the Hugging Face Hub",
    )
    upload_parser.add_argument(
        "--branch",
        type=None,
        help="If set, the upload will be made to a particular branch of the provided `repo-id`.",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="If the repository should be private.",
    )
    upload_parser.set_defaults(func=upload_kernels)

    lock_parser = subparsers.add_parser("lock", help="Lock kernel revisions")
    lock_parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    lock_parser.set_defaults(func=lock_kernels)

    # Add generate-readme subcommand parser
    generate_readme_parser = subparsers.add_parser(
        "generate-readme",
        help="Generate README snippets for a kernel's public functions",
    )
    generate_readme_parser.add_argument(
        "repo_id",
        type=str,
        help="The kernel repo ID (e.g., kernels-community/activation)",
    )
    generate_readme_parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The kernel revision (branch, tag, or commit SHA, defaults to 'main')",
    )
    generate_readme_parser.set_defaults(
        func=lambda args: generate_readme_for_kernel(
            repo_id=args.repo_id, revision=args.revision
        )
    )

    # Add init subcommand parser
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new kernel project structure",
    )
    init_parser.add_argument(
        "kernel_name",
        type=str,
        help="Name of the kernel (e.g., 'my-kernel')",
    )
    init_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the kernel project (defaults to current directory)",
    )
    init_parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the kernel (e.g., 'kernels-community/my-kernel')",
    )

    init_parser.set_defaults(func=init_kernel_project)

    args = parser.parse_args()
    args.func(args)


def download_kernels(args):
    lock_path = args.project_dir / "kernels.lock"

    if not lock_path.exists():
        print(f"No kernels.lock file found in: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.project_dir / "kernels.lock", "r") as f:
        lock_json = json.load(f)

    all_successful = True

    for kernel_lock_json in lock_json:
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        print(
            f"Downloading `{kernel_lock.repo_id}` at with SHA: {kernel_lock.sha}",
            file=sys.stderr,
        )
        if args.all_variants:
            install_kernel_all_variants(
                kernel_lock.repo_id, kernel_lock.sha, variant_locks=kernel_lock.variants
            )
        else:
            try:
                install_kernel(
                    kernel_lock.repo_id,
                    kernel_lock.sha,
                    variant_locks=kernel_lock.variants,
                )
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                all_successful = False

    if not all_successful:
        sys.exit(1)


def lock_kernels(args):
    with open(args.project_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)

    all_locks = []
    for kernel, version in kernel_versions.items():
        all_locks.append(get_kernel_locks(kernel, version))

    with open(args.project_dir / "kernels.lock", "w") as f:
        json.dump(all_locks, f, cls=_JSONEncoder, indent=2)


def upload_kernels(args):
    # Resolve `kernel_dir` to be uploaded.
    kernel_dir = Path(args.kernel_dir).resolve()

    build_dir = None
    for candidate in [kernel_dir / "build", kernel_dir]:
        variants = [
            variant_path
            for variant_path in candidate.glob("torch*")
            if BUILD_VARIANT_REGEX.match(variant_path.name) is not None
        ]
        if variants:
            build_dir = candidate
            break
    if build_dir is None:
        raise ValueError(
            f"Couldn't find any build variants in: {kernel_dir.absolute()} or {(kernel_dir / 'build').absolute()}"
        )

    repo_id = create_repo(
        repo_id=args.repo_id, private=args.private, exist_ok=True
    ).repo_id

    if args.branch is not None:
        create_branch(repo_id=repo_id, branch=args.branch, exist_ok=True)

    delete_patterns: set[str] = set()
    for build_variant in build_dir.iterdir():
        if build_variant.is_dir():
            delete_patterns.add(f"{build_variant.name}/**")

    upload_folder(
        repo_id=repo_id,
        folder_path=build_dir,
        revision=args.branch,
        path_in_repo="build",
        delete_patterns=list(delete_patterns),
        commit_message="Build uploaded using `kernels`.",
        allow_patterns=["torch-*"],
    )
    print(f"✅ Kernel upload successful. Find the kernel in https://hf.co/{repo_id}.")


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def check_kernel(
    *, macos: str, manylinux: str, python_abi: str, repo_id: str, revision: str
):
    try:
        import kernels.check
    except ImportError:
        print(
            "`kernels check` requires the `kernel-abi-check` package: pip install kernel-abi-check",
            file=sys.stderr,
        )
        sys.exit(1)

    kernels.check.check_kernel(
        macos=macos,
        manylinux=manylinux,
        python_abi=python_abi,
        repo_id=repo_id,
        revision=revision,
    )


def init_kernel_project(args):
    """Initialize a new kernel project with the standard structure."""
    kernel_name = args.kernel_name
    if "/" in kernel_name or "\\" in kernel_name:
        raise ValueError(
            "Kernel name cannot contain path separators, to specify an output directory use the --output-dir argument"
        )
    # Normalize kernel name (replace hyphens with underscores for Python compatibility)
    kernel_name_normalized = kernel_name.replace("-", "_")

    # Determine output directory
    if args.output_dir is not None:
        output_dir = (Path(args.output_dir) / kernel_name).resolve()
    else:
        output_dir = Path.cwd() / kernel_name

    # Determine repo_id
    repo_id = args.repo_id if args.repo_id else f"your-username/{kernel_name}"

    # Check if directory already exists
    if output_dir.exists() and any(output_dir.iterdir()):
        print(
            f"Error: Directory '{output_dir}' already exists and is not empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create directory structure
    dirs_to_create = [
        output_dir,
        output_dir / f"{kernel_name_normalized}_cuda",
        output_dir / f"{kernel_name_normalized}_rocm",
        output_dir / f"{kernel_name_normalized}_metal",
        output_dir / f"{kernel_name_normalized}_xpu",
        output_dir / "torch-ext",
        output_dir / "torch-ext" / kernel_name_normalized,
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Template substitution values
    template_values = {
        "kernel_name": kernel_name_normalized,
        "repo_id": repo_id,
    }

    # Create files
    files_to_create = {
        "flake.nix": DEFAULT_FLAKE_NIX % template_values,
        "build.toml": DEFAULT_BUILD_TOML % template_values,
        ".gitattributes": DEFAULT_GITATTRIBUTES,
        "README.md": DEFAULT_README % template_values,
        f"torch-ext/{kernel_name_normalized}/__init__.py": DEFAULT_INIT_PY
        % template_values,
        "torch-ext/torch_binding.cpp": DEFAULT_TORCH_BINDING_CPP % template_values,
        "torch-ext/torch_binding.h": DEFAULT_TORCH_BINDING_H % template_values,
    }

    for file_path, content in files_to_create.items():
        full_path = output_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)

    # Print success message
    print(
        f"✅ Kernel project '{kernel_name}' initialized successfully at: {output_dir}"
    )
    print()
    print("Project structure:")
    _print_tree(output_dir, prefix="")
    print()
    print("Next steps:")
    print(f"  1. cd {output_dir}")
    print(f"  2. Add your kernel implementation in {kernel_name_normalized}/")
    print(
        "  3. Update torch-ext/{kernel_name}/__init__.py to export your functions".format(
            kernel_name=kernel_name_normalized
        )
    )
    print("  4. Build with: nix run .#build-and-copy ")
    print(f"  5. Upload with: kernels upload . --repo-id {repo_id}")


def _print_tree(directory: Path, prefix: str = ""):
    """Print a directory tree structure."""
    entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    entries = [
        e
        for e in entries
        if not e.name.startswith(".git") or e.name == ".gitattributes"
    ]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _print_tree(entry, prefix + extension)
