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


def _set_top_level_version(doc, path: Path, new_version: str) -> str | None:
    """Update the top-level version in an already-parsed tomlkit doc.

    Returns the old version string, or *None* if it was already up-to-date.
    """
    table = _version_table(doc, path)
    old_version = str(table["version"])
    if old_version == new_version:
        return None
    table["version"] = new_version
    return old_version


def _set_path_dep_versions(doc, new_version: str) -> list[tuple[str, str]]:
    """Update the ``version`` field of every dependency that also has a ``path`` key.

    Path dependencies are local dependencies, but they need to have the correct version,
    otherwise `cargo publish` will croak.

    Returns a list of ``(dep_name, old_version)`` for each dependency that was
    changed.  Dependencies that only have a ``version`` (external crates.io deps)
    are left untouched.
    """
    changed: list[tuple[str, str]] = []
    for dep_name, dep_value in doc.get("dependencies", {}).items():
        if not isinstance(dep_value, dict):
            continue
        if "path" not in dep_value or "version" not in dep_value:
            continue
        old = str(dep_value["version"])
        if old == new_version:
            continue
        dep_value["version"] = new_version
        changed.append((dep_name, old))
    return changed


def replace_pyproject_version(
    path: Path, new_version: str, *, dry_run: bool
) -> str | None:
    """Rewrite the top-level version in a pyproject.toml file.

    Returns the old version string, or *None* if the file was already
    up-to-date (in which case the file is not written).
    """
    doc = tomlkit.parse(path.read_text())
    old_version = _set_top_level_version(doc, path, new_version)
    if old_version is not None and not dry_run:
        path.write_text(tomlkit.dumps(doc))
    return old_version


def replace_cargo_version(
    path: Path, new_version: str, *, dry_run: bool
) -> tuple[str | None, list[tuple[str, str]]]:
    """Rewrite the package version and all versioned path-dependency versions
    in a Cargo.toml file in a single parse/write pass.

    Returns ``(old_package_version | None, [(dep_name, old_dep_version), ...])``,
    where the first element is *None* when the package version was already
    up-to-date.  The file is written at most once, and only when something
    actually changed.
    """
    doc = tomlkit.parse(path.read_text())
    old_package = _set_top_level_version(doc, path, new_version)
    changed_deps = _set_path_dep_versions(doc, new_version)
    if (old_package is not None or changed_deps) and not dry_run:
        path.write_text(tomlkit.dumps(doc))
    return old_package, changed_deps


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")
