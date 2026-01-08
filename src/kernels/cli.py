import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from huggingface_hub import create_repo, upload_folder, create_branch

from kernels.compat import tomllib
from kernels.lockfile import KernelLock, get_kernel_locks
from kernels.utils import install_kernel, install_kernel_all_variants

from .doc import generate_readme_for_kernel

BUILD_VARIANT_REGEX = re.compile(r"^(torch\d+\d+|torch-(cpu|cuda|metal|rocm|xpu))")


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
        allow_patterns=["torch*"],
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
