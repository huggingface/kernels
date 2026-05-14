#!/usr/bin/env python3
"""Bump all version strings in the repo.

The two flags ``--dev`` and ``--major`` are independent and may be combined.

``--major`` controls *which component* is incremented:

  False (default): if currently a dev release, keep the same release base;
                   if currently a plain release, bump the patch component.
  True:            bump the minor component and reset patch to 0, regardless
                   of whether the current version is a dev release.

``--dev`` controls whether a dev suffix is added to the result:

  False (default): no dev suffix.
  True:            append ``.devN`` (Python) / ``-devN`` (Cargo).
                   If the target base matches the current base and the current
                   version is already a dev release, N is bumped by one;
                   otherwise N starts at 0.

Examples (codebase at ``0.10.1.dev0``)::

  (none)          -> 0.10.1
  --major         -> 0.11.0
  --dev           -> 0.10.1.dev1
  --dev --major   -> 0.11.0.dev0

Examples (codebase at ``0.10.1``)::

  (none)          -> 0.10.2
  --major         -> 0.11.0
  --dev           -> 0.10.2.dev0
  --dev --major   -> 0.11.0.dev0
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


def compute_versions(current: Version, *, dev: bool, major: bool) -> tuple[str, str]:
    """Return ``(python_version, cargo_version)`` for the requested bump."""
    if current.pre is not None or current.post is not None or current.local is not None:
        raise SystemExit(
            f"Codebase version `{current}` has pre/post/local components and "
            "cannot be bumped automatically. Set "
            f"{display_path(PRIMARY_PYPROJECT)} to a plain or dev release first "
            "(e.g. 0.10.1 or 0.10.1.dev0)."
        )

    release = current.release
    if len(release) < 2:
        raise SystemExit(
            f"Codebase version `{current}` is missing a minor component; "
            "expected at least MAJOR.MINOR.PATCH (e.g. 0.10.1)."
        )

    cur_major = release[0]
    cur_minor = release[1]
    cur_patch = release[2] if len(release) > 2 else 0
    cur_dev = current.dev  # int if dev release, else None

    # Determine the target base tuple (major, minor, patch).
    if major:
        base = (cur_major, cur_minor + 1, 0)
    elif cur_dev is not None:
        base = (cur_major, cur_minor, cur_patch)  # same base as current dev
    else:
        base = (cur_major, cur_minor, cur_patch + 1)  # bump patch from release

    base_str = f"{base[0]}.{base[1]}.{base[2]}"

    if dev:
        # Bump the dev counter when staying on the same release base; else 0.
        same_base = (
            not major
            and cur_dev is not None
            and base == (cur_major, cur_minor, cur_patch)
        )
        dev_num = (cur_dev + 1) if same_base else 0
        return f"{base_str}.dev{dev_num}", f"{base_str}-dev{dev_num}"

    return base_str, base_str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Add or bump the dev suffix on the result.",
    )
    parser.add_argument(
        "--major",
        action="store_true",
        help="Bump the minor component (and reset patch) instead of patch.",
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
    python_ver, cargo_ver = compute_versions(current, dev=args.dev, major=args.major)

    print(f"Codebase kernels version : {current}")
    if python_ver == cargo_ver:
        print(f"Next version             : {python_ver}")
        confirm_prompt = f"Bump all version sites to {python_ver}?"
    else:
        print(f"Next Python version      : {python_ver}")
        print(f"Next Cargo  version      : {cargo_ver}")
        confirm_prompt = f"Bump all version sites to {python_ver} / {cargo_ver}?"
    print()

    if args.dev and args.major:
        makefile_target = "bump-dev-major"
    elif args.dev:
        makefile_target = "bump-dev"
    elif args.major:
        makefile_target = "bump-major"
    else:
        makefile_target = "bump-release"

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
