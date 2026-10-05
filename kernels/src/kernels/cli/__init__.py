import argparse
import sys
from pathlib import Path
from typing import Literal

from kernels.cli.download import download_kernels
from kernels.cli.info import print_kernel_info
from kernels.cli.inspect import print_kernel_api
from kernels.cli.lock import lock_kernels
from kernels.cli.variants import print_kernel_variants
from kernels.cli.verify_signature import verify_signature
from kernels.cli.versions import print_kernel_versions


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

    inspect_parser = subparsers.add_parser("inspect", help="Show a kernel's functions and layers")
    inspect_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    inspect_selection = inspect_parser.add_mutually_exclusive_group(required=True)
    inspect_selection.add_argument(
        "--revision",
        type=str,
        help="The kernel revision (branch, tag, or commit). Cannot be used together with --version.",
    )
    inspect_selection.add_argument("--version", type=int, help="The kernel version")
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the functions and layers as JSON",
    )
    inspect_parser.set_defaults(func=kernel_inspect)

    variants_parser = subparsers.add_parser("variants", help="Show kernel build variants and compatibility decisions")
    variants_parser.add_argument("repo_id", type=str, help="The kernel repo ID")
    variants_selection = variants_parser.add_mutually_exclusive_group()
    variants_selection.add_argument(
        "--all-versions",
        action="store_true",
        help="Show variants for all versions (default: latest version only)",
    )
    variants_selection.add_argument(
        "--version", type=_variant_version, help="Show variants for an integer kernel version or 'latest'"
    )
    variants_selection.add_argument("--revision", type=str, help="Show variants for a branch, tag, or commit")
    variants_parser.add_argument(
        "--only-compatible", action="store_true", help="Only show variants compatible with the current system"
    )
    variants_parser.set_defaults(func=kernel_variants)

    versions_parser = subparsers.add_parser("versions", help="Deprecated: use `kernels variants --all-versions`")
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

    args = parser.parse_args()
    args.func(args)


def _variant_version(value: str) -> int | Literal["latest"]:
    if value == "latest":
        return "latest"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("version must be an integer or 'latest'") from None


def kernel_info(args):
    print_kernel_info(
        args.kernel,
        revision=args.revision,
        version=args.version,
        json_output=args.json,
    )


def kernel_inspect(args):
    print_kernel_api(
        args.repo_id,
        revision=args.revision,
        version=args.version,
        json_output=args.json,
    )


def kernel_versions(args):
    print_kernel_versions(args.repo_id)


def kernel_variants(args):
    print_kernel_variants(
        args.repo_id,
        all_versions=args.all_versions,
        only_compatible=args.only_compatible,
        version=args.version,
        revision=args.revision,
    )


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
