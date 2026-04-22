"""Shared helpers for bump_to_dev.py and pre_release.py.

Both scripts rewrite the same set of version sites; only the version-computation
step differs. Constants, I/O helpers, and the interactive prompt live here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_PYPROJECT = REPO_ROOT / "kernels" / "pyproject.toml"

PYPROJECT_FILES = [
    PRIMARY_PYPROJECT
]

CARGO_FILES = [
    REPO_ROOT / "kernels-data" / "Cargo.toml",
    REPO_ROOT / "kernels-data" / "bindings" / "python" / "Cargo.toml",
    REPO_ROOT / "kernel-builder" / "Cargo.toml",
    REPO_ROOT / "kernel-abi-check" / "kernel-abi-check" / "Cargo.toml",
    REPO_ROOT / "kernel-abi-check" / "bindings" / "python" / "Cargo.toml",
]


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def _parsed_version(path: Path) -> str:
    """Return the top-level version from a pyproject.toml or Cargo.toml.

    pyproject.toml stores it under ``[project]``; Cargo.toml under ``[package]``.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    for table in ("project", "package"):
        section = data.get(table)
        if isinstance(section, dict) and "version" in section:
            return section["version"]

    raise SystemExit(
        f"Could not find a top-level `version` under [project] or [package] in {display_path(path)}."
    )


def get_codebase_version() -> str:
    return _parsed_version(PRIMARY_PYPROJECT)


def replace_top_level_version(path: Path, new_version: str, *, dry_run: bool) -> str | None:
    old_version = _parsed_version(path)
    if old_version == new_version:
        return None

    # Writing TOML without pulling in tomlkit: re-emit the exact `version = "<old>"`
    # line we just parsed. Anchoring to ``^`` plus the escaped old value makes this
    # a single, unambiguous match — inline dep entries can't collide.
    original = path.read_text()
    pattern = re.compile(
        r'^(version\s*=\s*)"' + re.escape(old_version) + r'"',
        re.MULTILINE,
    )
    updated, count = pattern.subn(rf'\1"{new_version}"', original, count=1)
    if count != 1:
        raise SystemExit(
            f"Parsed version `{old_version}` from {display_path(path)} but could not locate "
            "the matching top-level `version = \"...\"` line to rewrite."
        )

    if not dry_run:
        path.write_text(updated)
    return old_version


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")
