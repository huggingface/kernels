"""
Refresh the vendored ABI symbol files used by `kernel-abi-check`.

Use `--check` to only report whether any file is outdated, without writing
changes (exit status `1` if something is stale). This is handy for a local
sanity check.
"""

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SymbolFile:
    """An upstream-maintained file vendored into `kernel-abi-check`."""

    name: str
    url: str
    # Destination relative to the repository root.
    dest: str


SYMBOL_FILES = [
    SymbolFile(
        name="CPython stable ABI",
        url="https://raw.githubusercontent.com/python/cpython/main/Misc/stable_abi.toml",
        dest="kernel-abi-check/src/python_abi/stable_abi.toml",
    ),
    SymbolFile(
        name="manylinux policy",
        url="https://raw.githubusercontent.com/pypa/auditwheel/main/src/auditwheel/policy/manylinux-policy.json",
        dest="kernel-abi-check/src/manylinux/manylinux-policy.json",
    ),
    SymbolFile(
        name="Torch shim function versions",
        url="https://raw.githubusercontent.com/pytorch/pytorch/refs/heads/main/torch/csrc/stable/c/shim_function_versions.txt",
        dest="kernel-abi-check/src/torch_stable_abi/shim_function_versions.txt",
    ),
]


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 (trusted URLs)
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report stale files, do not write changes (exit 1 if stale)",
    )
    args = parser.parse_args()

    changed = []
    for symbol_file in SYMBOL_FILES:
        dest = REPO_ROOT / symbol_file.dest
        latest = fetch(symbol_file.url)
        current = dest.read_bytes() if dest.exists() else None

        if latest == current:
            print(f"up-to-date: {symbol_file.name} ({symbol_file.dest})")
            continue

        changed.append(symbol_file)
        if args.check:
            print(f"OUTDATED:   {symbol_file.name} ({symbol_file.dest})")
        else:
            dest.write_bytes(latest)
            print(f"updated:    {symbol_file.name} ({symbol_file.dest})")

    if not changed:
        print("\nAll ABI symbol files are up-to-date.")
        return 0

    if args.check:
        print(f"\n{len(changed)} ABI symbol file(s) are outdated.")
        return 1

    print(f"\nUpdated {len(changed)} ABI symbol file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
