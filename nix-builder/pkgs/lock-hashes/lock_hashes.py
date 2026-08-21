#!/usr/bin/env python3
"""Enrich a kernel lock file with Nix hashes.

Outputs the lock file as-is, with a `hash` field added to every lock. The hash
is the SRI hash of a snapshot of the kernel repository, as used by the
`fetchFromHuggingFace` fixed-output derivation.

Must be run from the `kernel-builder` dev shell, which provides `kernels-data`
and `huggingface_hub`.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from kernels_data import KernelLocks, NixKernelLock, NixKernelLocks


def nix_hash(repo_id: str, revision: str) -> str:
    """Download a kernel repository and return the SRI hash of the snapshot."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot = Path(tmp_dir) / "snapshot"
        subprocess.run(
            [
                sys.executable,
                str("@download@"),
                repo_id,
                "kernel",
                revision,
                str(snapshot),
            ],
            check=True,
        )
        # `--type sha256 --sri` matches `outputHashMode = "recursive"`, which
        # hashes the NAR serialization of the snapshot.
        hash = subprocess.run(
            ["nix", "hash", "path", "--type", "sha256", "--sri", str(snapshot)],
            check=True,
            capture_output=True,
            text=True,
        )

    return hash.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "lockfile",
        type=Path,
        nargs="?",
        help="kernel lock file to enrich, defaults to standard input",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="file to write to, defaults to standard output",
    )
    args = parser.parse_args()

    if args.lockfile is None:
        locks = KernelLocks.from_json(sys.stdin.read())
    else:
        locks = KernelLocks.from_json(args.lockfile.read_text())

    nix_locks = NixKernelLocks(
        {
            dep: NixKernelLock(lock.commit, nix_hash(dep.repo_id, lock.commit))
            for dep, lock in _report(locks.items())
        }
    )

    if args.output is None:
        print(nix_locks.to_json())
    else:
        args.output.write_text(nix_locks.to_json() + "\n")


def _report(items):
    """Report progress on standard error, so that standard output stays clean."""
    for n, (dep, lock) in enumerate(items, start=1):
        print(
            f"[{n}/{len(items)}] Hashing {dep.repo_id} at {lock.commit}",
            file=sys.stderr,
        )
        yield dep, lock


if __name__ == "__main__":
    main()
