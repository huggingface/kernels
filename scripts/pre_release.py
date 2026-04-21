#!/usr/bin/env python3
"""Strip the development suffix from all version strings in the repo.

Reads the current ``kernels`` version from ``kernels/pyproject.toml`` (the
source-of-truth in the codebase — no install required) and, assuming it is a
development version like ``0.14.0.dev0``, rewrites every version site to the
corresponding release: ``0.14.0``. The inverse of ``bump_to_dev.py``.

Example: codebase at ``0.14.0.dev0`` -> all sites become ``0.14.0``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from _version_common import (
    CARGO_FILES,
    PRIMARY_PYPROJECT,
    PYPROJECT_FILES,
    confirm,
    display_path,
    get_codebase_version,
    replace_top_level_version,
)

# Accept PEP 440 dev versions like `0.14.0.dev0` (the form kernels/pyproject.toml
# uses). Cargo.toml files use `0.14.0-dev0`, but we never read from them — we
# only write to them, using the release string we derive from pyproject.
DEV_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?\.dev\d+$")


def next_release_version(current: str) -> str:
    match = DEV_VERSION_RE.match(current.strip())
    if match is None:
        raise SystemExit(
            f"Codebase version `{current}` is not a development version "
            "(e.g. 0.14.0.dev0). This tool strips the dev suffix ahead of a "
            f"release. Set {display_path(PRIMARY_PYPROJECT)} to a dev version first."
        )

    major, minor = int(match.group(1)), int(match.group(2))
    patch = int(match.group(3)) if match.group(3) is not None else 0
    return f"{major}.{minor}.{patch}"


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
    release = next_release_version(current)

    print(f"Codebase kernels version : {current}")
    print(f"Next release version     : {release}")
    print()

    if not args.dry_run and not args.yes:
        if not confirm(f"Strip dev suffix from all version sites -> {release}?"):
            print("Aborted; no files changed.")
            return 1
        print()

    changed: list[tuple[Path, str, str]] = []
    for path in PYPROJECT_FILES:
        old = replace_top_level_version(path, release, dry_run=args.dry_run)
        if old is not None:
            changed.append((path, old, release))
    for path in CARGO_FILES:
        old = replace_top_level_version(path, release, dry_run=args.dry_run)
        if old is not None:
            changed.append((path, old, release))

    verb = "Would update" if args.dry_run else "Updated"
    if not changed:
        print("All files already at the target version; nothing to do.")
        return 0

    print(f"{verb} {len(changed)} file(s):")
    for path, old, new in changed:
        print(f"  {display_path(path)}: {old} -> {new}")

    if not args.dry_run:
        print()
        print("Note: Cargo.lock and kernels/uv.lock are refreshed by the `pre-release`")
        print("Makefile target; if you ran this script directly, regenerate them with")
        print("`cargo check --workspace` and `(cd kernels && uv lock)`.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
