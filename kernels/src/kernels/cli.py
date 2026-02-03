import argparse
import dataclasses
import json
import sys
from pathlib import Path

from kernels.compat import tomllib
from kernels.lockfile import KernelLock, get_kernel_locks
from kernels.upload import upload_kernels_dir
from kernels.utils import install_kernel, install_kernel_all_variants
from kernels.versions_cli import print_kernel_versions

from .doc import generate_readme_for_kernel


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

    versions_parser = subparsers.add_parser("versions", help="Show kernel versions")
    versions_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    versions_parser.set_defaults(func=kernel_versions)

    upload_parser = subparsers.add_parser("upload", help="Upload kernels to the Hub")
    upload_parser.add_argument(
        "kernel_dir",
        type=Path,
        help="Directory of the kernel build",
    )
    upload_parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID to use to upload to the Hugging Face Hub",
    )
    upload_parser.add_argument(
        "--branch",
        type=str,
        default=None,
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

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run and submit benchmark results for a kernel",
    )
    benchmark_parser.add_argument(
        "repo_id",
        type=str,
        help="Kernel repo ID (e.g., kernels-community/activation)",
    )
    benchmark_parser.add_argument(
        "--branch", type=str, help="Kernel branch to benchmark"
    )
    benchmark_parser.add_argument(
        "--version", type=int, help="Kernel version to benchmark"
    )
    benchmark_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save JSON results to file",
    )
    benchmark_parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON results to stdout (in addition to table)",
    )
    benchmark_parser.add_argument("--iterations", type=int, default=100)
    benchmark_parser.add_argument("--warmup", type=int, default=10)
    benchmark_parser.set_defaults(func=run_benchmark)

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


def kernel_versions(args):
    print_kernel_versions(args.repo_id)


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
    upload_kernels_dir(
        Path(args.kernel_dir).resolve(),
        repo_id=args.repo_id,
        branch=args.branch,
        private=args.private,
    )


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


def run_benchmark(args):
    from kernels import benchmark

    benchmark.run_benchmark(
        repo_id=args.repo_id,
        branch=args.branch,
        version=args.version,
        iterations=args.iterations,
        warmup=args.warmup,
        output=args.output,
        print_json=args.json,
    )
