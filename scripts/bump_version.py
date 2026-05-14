#!/usr/bin/env python3
"""Bump all version strings in the repo.

Without ``--dev``: strip the development suffix ahead of a release.
  Example: codebase at ``0.14.0.dev0`` -> all sites become ``0.14.0``.

With ``--dev``: advance to the next development cycle.
  Example: codebase at ``0.13.0`` -> Python sites get ``0.14.0.dev0`` (PEP 440),
  Cargo sites get ``0.14.0-dev0``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _version_common import (
    CARGO_FILES,
    PRIMARY_PYPROJECT,
    PYPROJECT_FILES,
    confirm,
    display_path,
    get_codebase_version,
    replace_cargo_version,
    replace_pyproject_version,
)
from packaging.version import Version


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


def next_release_version(current: Version) -> tuple[str, str]:
    if (
        not current.is_devrelease
        or current.pre is not None
        or current.post is not None
        or current.local is not None
    ):
        raise SystemExit(
            f"Codebase version `{current}` is not a development version "
            "(e.g. 0.14.0.dev0). This tool strips the dev suffix ahead of a "
            f"release. Set {display_path(PRIMARY_PYPROJECT)} to a dev version first."
        )

    release = current.release
    major = release[0] if len(release) > 0 else 0
    minor = release[1] if len(release) > 1 else 0
    patch = release[2] if len(release) > 2 else 0
    ver = f"{major}.{minor}.{patch}"
    return ver, ver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Bump to the next dev cycle.",
    )
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

    if args.dev:
        python_ver, cargo_ver = next_dev_versions(current)
        print(f"Codebase kernels version : {current}")
        print(f"Next Python dev version  : {python_ver}")
        print(f"Next Cargo  dev version  : {cargo_ver}")
        confirm_prompt = f"Bump all version sites to {python_ver} / {cargo_ver}?"
        makefile_target = "bump-dev"
    else:
        python_ver, cargo_ver = next_release_version(current)
        print(f"Codebase kernels version : {current}")
        print(f"Next release version     : {python_ver}")
        confirm_prompt = f"Strip dev suffix from all version sites -> {python_ver}?"
        makefile_target = "bump-release"
    print()

    if not args.dry_run and not args.yes:
        if not confirm(confirm_prompt):
            print("Aborted; no files changed.")
            return 1
        print()

    changed: list[tuple[Path, str, str]] = []
    for path in PYPROJECT_FILES:
        old = replace_pyproject_version(path, python_ver, dry_run=args.dry_run)
        if old is not None:
            changed.append((path, old, python_ver))
    for path in CARGO_FILES:
        old_pkg, old_deps = replace_cargo_version(path, cargo_ver, dry_run=args.dry_run)
        if old_pkg is not None:
            changed.append((path, old_pkg, cargo_ver))
        for dep_name, old_dep in old_deps:
            changed.append(
                (path, f"Path dependency `{dep_name}`: {old_dep}", cargo_ver)
            )

    verb = "Would update" if args.dry_run else "Updated"
    if not changed:
        print("All files already at the target version; nothing to do.")
        return 0

    print(f"{verb} {len(changed)} file(s):")
    for path, old, new in changed:
        print(f"  {display_path(path)}: {old} -> {new}")

    if not args.dry_run:
        print()
        print(
            f"Note: Cargo.lock and kernels/uv.lock are refreshed by the `{makefile_target}`"
        )
        print("Makefile target; if you ran this script directly, regenerate them with")
        print("`cargo check --workspace` and `(cd kernels && uv lock)`.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
