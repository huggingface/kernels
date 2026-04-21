"""Shared helpers for bump_to_dev.py and pre_release.py.

Both scripts rewrite the same set of version sites; only the version-computation
step differs. Constants, I/O helpers, and the interactive prompt live here.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_PYPROJECT = REPO_ROOT / "kernels" / "pyproject.toml"

PYPROJECT_FILES = [
    PRIMARY_PYPROJECT,
]

CARGO_FILES = [
    REPO_ROOT / "kernels-data" / "Cargo.toml",
    REPO_ROOT / "kernels-data" / "bindings" / "python" / "Cargo.toml",
    REPO_ROOT / "kernel-builder" / "Cargo.toml",
    REPO_ROOT / "kernel-abi-check" / "kernel-abi-check" / "Cargo.toml",
    REPO_ROOT / "kernel-abi-check" / "bindings" / "python" / "Cargo.toml",
]

# Anchored to the start of a line so inline dep entries like
# `clap = { version = "4", ... }` are never matched.
VERSION_LINE_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def get_codebase_version() -> str:
    text = PRIMARY_PYPROJECT.read_text()
    match = VERSION_LINE_RE.search(text)
    if match is None:
        raise SystemExit(f"Could not find a top-level `version = \"...\"` line in {PRIMARY_PYPROJECT}.")
    return match.group(1)


def replace_top_level_version(path: Path, new_version: str, *, dry_run: bool) -> str | None:
    original = path.read_text()
    match = VERSION_LINE_RE.search(original)
    if match is None:
        raise SystemExit(f"Could not find a top-level `version = \"...\"` line in {path}.")

    old_version = match.group(1)
    if old_version == new_version:
        return None

    updated = VERSION_LINE_RE.sub(f'version = "{new_version}"', original, count=1)
    if not dry_run:
        path.write_text(updated)
    return old_version


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")
