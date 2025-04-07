import argparse
import dataclasses
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from kernels.compat import tomllib
from kernels.lockfile import KernelLock, get_kernel_locks
from kernels.utils import install_kernel, install_kernel_all_variants


def main():
    parser = argparse.ArgumentParser(
        prog="kernel", description="Manage compute kernels"
    )
    subparsers = parser.add_subparsers(required=True)

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

    lock_parser = subparsers.add_parser("lock", help="Lock kernel revisions")
    lock_parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    lock_parser.set_defaults(func=lock_kernels)

    # Add a new compatibility command
    compat_parser = subparsers.add_parser(
        "compatibility", help="Show kernel build compatibility"
    )
    compat_parser.add_argument(
        "repo_id",
        type=str,
        help="The repository ID of the kernel (e.g., 'kernels-community/activation')",
    )
    compat_parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The revision of the kernel (default: main)",
    )
    compat_parser.set_defaults(func=check_compatibility)

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


def check_compatibility(args):
    """Check build compatibility for a kernel by reading its build.toml file."""
    try:
        # Download only the build.toml file from the repository
        build_toml_path = hf_hub_download(
            repo_id=args.repo_id,
            filename="build.toml",
            revision=args.revision,
        )
    except Exception:
        print(
            f"Error: Could not find build.toml in repository {args.repo_id}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse the build.toml file
    try:
        with open(build_toml_path, "rb") as f:
            content = f.read().decode("utf-8")

        # Simple check for compatibility without full parsing
        is_universal = "language" in content and "python" in content
        has_cuda = "cuda-capabilities" in content
        has_rocm = "rocm-archs" in content

    except Exception as e:
        print(f"Error reading build.toml: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Print the compatibility
    print(f"Kernel: {args.repo_id}")
    print("Compatibility: ", end="")

    if is_universal:
        print("universal")
    else:
        compatibilities = []
        if has_cuda:
            compatibilities.append("cuda")
        if has_rocm:
            compatibilities.append("rocm")
        print(", ".join(compatibilities) if compatibilities else "unknown")

    return 0


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
