import argparse
import dataclasses
import json
import sys
from pathlib import Path

from hf_kernels.compat import tomllib
from hf_kernels.lockfile import KernelLock, get_kernel_locks
from hf_kernels.utils import install_kernel, install_kernel_all_variants


def main():
    parser = argparse.ArgumentParser(
        prog="hf-kernel", description="Manage compute kernels"
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

    args = parser.parse_args()
    args.func(args)


def download_kernels(args):
    lock_path = args.project_dir / "hf-kernels.lock"

    if not lock_path.exists():
        print(f"No hf-kernels.lock file found in: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.project_dir / "hf-kernels.lock", "r") as f:
        lock_json = json.load(f)

    for kernel_lock_json in lock_json:
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        print(
            f"Downloading `{kernel_lock.repo_id}` at with SHA: {kernel_lock.sha}",
            file=sys.stderr,
        )
        if args.all_variants:
            install_kernel_all_variants(kernel_lock.repo_id, kernel_lock.sha)
        else:
            install_kernel(kernel_lock.repo_id, kernel_lock.sha)


def lock_kernels(args):
    with open(args.project_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)

    all_locks = []
    for kernel, version in kernel_versions.items():
        all_locks.append(get_kernel_locks(kernel, version))

    with open(args.project_dir / "hf-kernels.lock", "w") as f:
        json.dump(all_locks, f, cls=_JSONEncoder, indent=2)


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
