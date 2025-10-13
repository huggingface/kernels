import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from kernel_abi_check import (
    BinaryFormat,
    IncompatibleAbi3Symbol,
    IncompatibleMacOSVersion,
    IncompatibleManylinuxSymbol,
    MissingMacOSVersion,
    NonAbi3Symbol,
    ObjectFile,
)

from kernels.utils import CACHE_DIR


def check_kernel(
    *, macos: str, manylinux: str, python_abi: str, repo_id: str, revision: str
):
    variants_path = (
        Path(
            snapshot_download(
                repo_id,
                allow_patterns=["build/*"],
                cache_dir=CACHE_DIR,
                revision=revision,
            )
        )
        / "build"
    )

    has_issues = False
    for variant_path in variants_path.iterdir():
        if not variant_path.is_dir():
            print(
                f"⛔ `build/` must only contain directories, found: {variant_path.name}",
                file=sys.stderr,
            )
            has_issues = True
            continue

        print(f"Checking variant: {variant_path.name}", file=sys.stderr)

        indent = 2

        for dylib_path in variant_path.rglob("*.so"):
            print_with_indent(
                indent,
                f"Dynamic library {dylib_path.relative_to(variant_path)}:",
            )

            o = ObjectFile(dylib_path)
            has_issues |= check_abi3(o, python_abi, indent + 2)

            # TODO: also check operating system
            if o.format() == BinaryFormat.ELF:
                has_issues |= check_manylinux(o, manylinux, indent + 2)
            elif o.format() == BinaryFormat.MACH_O:
                has_issues |= check_macos(o, macos, indent + 2)

    if has_issues:
        sys.exit(1)


def check_abi3(object_file: ObjectFile, python_abi: str, indent: int) -> bool:
    has_issues = False
    violations = object_file.check_python_abi(python_abi)
    if violations != []:
        has_issues = True
        print_with_indent(
            indent,
            f"⛔ Found symbols that are incompatible with Python ABI {python_abi}:",
        )
        for violation in violations:
            if isinstance(violation, IncompatibleAbi3Symbol):
                print_with_indent(
                    indent + 3,
                    f"{violation.name}: {violation.version_added}",
                )
            elif isinstance(violation, NonAbi3Symbol):
                print_with_indent(
                    indent + 3,
                    f"{violation.name}",
                )
    else:
        print_with_indent(indent, f"🐍 Python ABI {python_abi} compatible")

    return has_issues


def check_macos(object_file: ObjectFile, macos: str, indent: int) -> bool:
    has_issues = False
    violations = object_file.check_macos(macos)
    if violations != []:
        has_issues = True
        print_with_indent(
            indent,
            f"⛔ Found incompatibility with macOS {macos}:",
        )

        for violation in violations:
            if isinstance(violation, MissingMacOSVersion):
                print_with_indent(
                    indent + 3,
                    "shared library does not contain macOS version",
                )
            elif isinstance(violation, IncompatibleMacOSVersion):
                print_with_indent(
                    indent + 3,
                    f"shared library requires macOS {violation.version}",
                )
    else:
        print_with_indent(indent, f"🍏 compatible with macOS {macos}")

    return has_issues


def check_manylinux(object_file: ObjectFile, manylinux: str, indent: int) -> bool:
    has_issues = False
    violations = object_file.check_manylinux(manylinux)
    if violations != []:
        has_issues = True
        print_with_indent(
            indent,
            f"⛔ Found symbols that are incompatible with {manylinux}:",
        )

        for violation in violations:
            if isinstance(violation, IncompatibleManylinuxSymbol):
                print_with_indent(
                    indent + 3,
                    f"{violation.name}_{violation.dep}: {violation.version}",
                )
    else:
        print_with_indent(indent, f"🐧 {manylinux} compatible")

    return has_issues


def print_with_indent(indent: int, message: str):
    print(f"{' ' * indent}{message}", file=sys.stderr)
