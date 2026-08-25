#!/usr/bin/env python3
"""Lock a kernel's dependencies and compute the Nix hashes.

Reads the kernel's `build.toml`, resolves its dependencies to commits, and
outputs a lock file with a `hash` field added to every lock. The hash is the
SRI hash of a snapshot of the kernel repository, as used by the
`fetchFromHuggingFace` fixed-output derivation.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub.hf_api import HfApi
from kernels.locking import extract_dependency_locks
from kernels_data import Build, KernelLocks, NixKernelLock, NixKernelLocks


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


def lock_kernel_deps(kernel_dir: Path) -> KernelLocks:
    """Resolve the dependencies of the kernel in `kernel_dir` to commits."""
    build_toml = kernel_dir / "build.toml"
    if not build_toml.exists():
        raise FileNotFoundError(f"build.toml not found in {kernel_dir}")

    build = Build.open(kernel_dir)
    api = HfApi()

    kernel_locks = {}

    for backend in build.general.backends:
        for dep, lock in extract_dependency_locks(
            build.all_kernel_depends(backend), api=api, backend=str(backend)
        ).items():
            kernel_locks[dep] = lock

    return KernelLocks(locks=kernel_locks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "kernel_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="kernel directory to lock dependencies for, defaults to the current directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="file to write to, defaults to standard output",
    )
    args = parser.parse_args()

    locks = lock_kernel_deps(args.kernel_dir)

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
