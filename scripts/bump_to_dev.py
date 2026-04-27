#!/usr/bin/env python3
"""Bump all version strings in the repo to the next development version.

Reads the current ``kernels`` version from ``kernels/pyproject.toml`` (the
source-of-truth in the codebase — no install required).

Example: codebase at ``0.13.0`` -> Python sites get ``0.14.0.dev0`` (PEP 440)
and Cargo sites get ``0.14.0-dev0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.version import Version

from _version_common import (
    CARGO_FILES,
    PRIMARY_PYPROJECT,
    PYPROJECT_FILES,
    confirm,
    display_path,
    get_codebase_version,
    replace_top_level_version,
)


def next_dev_versions(current: Version) -> tuple[str, str]:
    if current.is_prerelease or current.is_postrelease or current.local is not None:
        raise SystemExit(
            f"Codebase version `{current}` is not a plain release (e.g. 0.13.0). "
            "This tool bumps from a release to the next dev cycle. Set "
            f"{display_path(PRIMARY_PYPROJECT)} to a release version first."
        )

    release = current.release
    if len(release) < 2:
        raise SystemExit(
            f"Codebase version `{current}` is missing a minor component; "
            "expected at least MAJOR.MINOR (e.g. 0.13.0)."
        )

    major, minor = release[0], release[1]
    next_minor = f"{major}.{minor + 1}.0"
    return f"{next_minor}.dev0", f"{next_minor}-dev0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without writing them.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args(argv)

    current = get_codebase_version()
    python_dev, cargo_dev = next_dev_versions(current)

    print(f"Codebase kernels version : {current}")
    print(f"Next Python dev version  : {python_dev}")
    print(f"Next Cargo  dev version  : {cargo_dev}")
    print()

    if not args.dry_run and not args.yes:
        if not confirm(f"Bump all version sites to {python_dev} / {cargo_dev}?"):
            print("Aborted; no files changed.")
            return 1
        print()

    changed: list[tuple[Path, str, str]] = []
    for path in PYPROJECT_FILES:
        old = replace_top_level_version(path, python_dev, dry_run=args.dry_run)
        if old is not None:
            changed.append((path, old, python_dev))
    for path in CARGO_FILES:
        old = replace_top_level_version(path, cargo_dev, dry_run=args.dry_run)
        if old is not None:
            changed.append((path, old, cargo_dev))

    verb = "Would update" if args.dry_run else "Updated"
    if not changed:
        print("All files already at the target version; nothing to do.")
        return 0

    print(f"{verb} {len(changed)} file(s):")
    for path, old, new in changed:
        print(f"  {display_path(path)}: {old} -> {new}")

    if not args.dry_run:
        print()
        print("Note: Cargo.lock and kernels/uv.lock are refreshed by the `bump-dev`")
        print("Makefile target; if you ran this script directly, regenerate them with")
        print("`cargo check --workspace` and `(cd kernels && uv lock)`.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
