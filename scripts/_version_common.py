"""Shared helpers for bump_to_dev.py and pre_release.py.

Both scripts rewrite the same set of version sites; only the version-computation
step differs. Constants, I/O helpers, and the interactive prompt live here.
"""

from __future__ import annotations

from pathlib import Path

import tomlkit
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent

PRIMARY_PYPROJECT = REPO_ROOT / "kernels" / "pyproject.toml"

PYPROJECT_FILES = [PRIMARY_PYPROJECT]

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


def _version_table(doc, path: Path):
    """Return the [project] or [package] table holding the top-level ``version``.

    pyproject.toml stores it under ``[project]``; Cargo.toml under ``[package]``.
    """
    for name in ("project", "package"):
        table = doc.get(name)
        if table is not None and "version" in table:
            return table
    raise SystemExit(
        f"Could not find a top-level `version` under [project] or [package] in {display_path(path)}."
    )


def get_codebase_version() -> Version:
    doc = tomlkit.parse(PRIMARY_PYPROJECT.read_text())
    return Version(str(_version_table(doc, PRIMARY_PYPROJECT)["version"]))


def replace_top_level_version(path: Path, new_version: str, *, dry_run: bool) -> str | None:
    doc = tomlkit.parse(path.read_text())
    table = _version_table(doc, path)

    old_version = str(table["version"])
    if old_version == new_version:
        return None

    table["version"] = new_version
    if not dry_run:
        path.write_text(tomlkit.dumps(doc))
    return old_version


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")
