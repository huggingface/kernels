import argparse
import sys
from pathlib import Path

from kernels_data import KernelDependency, KernelLocks, KernelVersion

from kernels.cli.info import print_kernel_info
from kernels.cli.lock_kernel_depends import print_lock_kernel_depends
from kernels.cli.verify_signature import verify_signature
from kernels.cli.versions import print_kernel_versions
from kernels.compat import tomllib
from kernels.hf_hub import _get_hf_api
from kernels.install import (
    install_kernel,
    install_kernel_all_variants,
)
from kernels.locking import extract_dependency_locks


def main():
    parser = argparse.ArgumentParser(prog="kernel", description="Manage compute kernels")
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser("check", help="Check a kernel for compliance")
    check_parser.add_argument("repo_id", type=str, nargs="?")
    check_parser.set_defaults(func=_check_moved)

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

    info_parser = subparsers.add_parser("info", help="Describe a kernel")
    info_parser.add_argument(
        "kernel",
        type=str,
        help="The kernel repo ID or a local path to a kernel repository",
    )
    info_parser.add_argument(
        "--revision",
        type=str,
        help="The kernel revision (branch, tag, or commit). Cannot be used together with --version.",
    )
    info_parser.add_argument(
        "--version",
        type=int,
        help="The kernel version. Cannot be used together with --revision.",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the kernel information as JSON",
    )
    info_parser.set_defaults(func=kernel_info)

    versions_parser = subparsers.add_parser("versions", help="Show kernel versions")
    versions_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    versions_parser.set_defaults(func=kernel_versions)

    lock_parser = subparsers.add_parser("lock", help="Lock kernel revisions")
    lock_parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    lock_parser.set_defaults(func=lock_kernels)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run and submit benchmark results for a kernel",
    )
    benchmark_parser.add_argument(
        "repo_id",
        type=str,
        help="Kernel repo ID (e.g., kernels-community/activation)",
    )
    benchmark_parser.add_argument("--branch", type=str, help="Kernel branch to benchmark")
    benchmark_parser.add_argument("--version", type=int, help="Kernel version to benchmark")
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
    benchmark_parser.add_argument(
        "--visual",
        type=str,
        default=None,
        help="Save visual outputs using this base path (e.g., --visual bench creates bench_light.svg and bench_dark.svg variants)",
    )
    benchmark_parser.add_argument(
        "--rasterized",
        action="store_true",
        help="Output PNG and GIF formats instead of SVG",
    )
    benchmark_parser.set_defaults(func=run_benchmark)

    verify_signature_parser = subparsers.add_parser(
        "verify-signature",
        help="Verify the signature of a kernel",
    )
    verify_signature_parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Download all build variants of the kernel",
    )
    verify_signature_parser.add_argument("--filter-unsigned", action="store_true", help="Skip unsigned variants")
    verify_signature_parser.add_argument(
        "--filter-no-digest",
        action="store_true",
        help="Skip variants without a digest in the metadata",
    )
    verify_signature_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    verify_signature_parser.add_argument("version", type=int, help="Kernel version to verify")
    verify_signature_parser.set_defaults(func=verify_signature)

    lock_kernel_depends_parser = subparsers.add_parser(
        "lock-kernel-depends",
        help="Print Nix dependencies for a project",
    )
    lock_kernel_depends_parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    lock_kernel_depends_parser.set_defaults(func=lock_kernel_depends)

    args = parser.parse_args()
    args.func(args)


def download_kernels(args):
    lock_path = args.project_dir / "kernels.lock"

    if not lock_path.exists():
        print(f"No kernels.lock file found in: {args.project_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.project_dir / "kernels.lock", "r") as f:
        kernel_locks = KernelLocks.from_json(f.read())

    all_successful = True

    # TODO: recurse into locks to also fetch dependencies.
    for _, kernel_lock in kernel_locks.locks.items():
        print(
            f"Downloading `{kernel_lock.repo_id}` with revision: {kernel_lock.revision}",
            file=sys.stderr,
        )
        if args.all_variants:
            install_kernel_all_variants(
                kernel_lock.repo_id,
                revision=kernel_lock.revision,
            )
        else:
            try:
                install_kernel(
                    kernel_lock.repo_id,
                    revision=kernel_lock.revision,
                )
            except FileNotFoundError as e:
                print(e, file=sys.stderr)
                all_successful = False

    if not all_successful:
        sys.exit(1)


def kernel_info(args):
    print_kernel_info(
        args.kernel,
        revision=args.revision,
        version=args.version,
        json_output=args.json,
    )


def kernel_versions(args):
    print_kernel_versions(args.repo_id)


def lock_kernel_depends(args):
    print_lock_kernel_depends(args.project_dir)


def lock_kernels(args):
    with open(args.project_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)

    depends = [
        KernelDependency(repo_id=kernel, version=KernelVersion.Version(version))
        for kernel, version in kernel_versions.items()
    ]

    locks = extract_dependency_locks(depends, api=_get_hf_api(), backend=None)

    print(locks)

    with open(args.project_dir / "kernels.lock", "w") as f:
        f.write(locks.to_json())


def _check_moved(_args):
    print(
        "`kernels check` has moved to `kernel-builder check-abi`",
        file=sys.stderr,
    )
    sys.exit(1)


def run_benchmark(args):
    from kernels.cli import benchmark

    benchmark.run_benchmark(
        repo_id=args.repo_id,
        branch=args.branch,
        version=args.version,
        iterations=args.iterations,
        warmup=args.warmup,
        output=args.output,
        print_json=args.json,
        visual=args.visual,
        rasterized=args.rasterized,
    )
