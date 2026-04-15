#!/usr/bin/env python3
"""Install kernels skills for AI coding assistants (Claude, Codex, OpenCode)."""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub.utils import get_session

DEFAULT_SKILL_ID = "cuda-kernels"
_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/huggingface/kernels/main/"
    "kernel-builder/skills/cuda-kernels"
)
_MANIFEST_URL = f"{_GITHUB_RAW_BASE}/manifest.txt"
_LOCAL_SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills" / "cuda-kernels"

GLOBAL_TARGETS = {
    "codex": Path("~/.codex/skills"),
    "claude": Path("~/.claude/skills"),
    "opencode": Path("~/.config/opencode/skills"),
}

LOCAL_TARGETS = {
    "codex": Path(".codex/skills"),
    "claude": Path(".claude/skills"),
    "opencode": Path(".opencode/skills"),
}


def _download(url: str) -> str:
    response = get_session().get(url)
    response.raise_for_status()
    return response.text


def _download_manifest() -> list[str]:
    entries: list[str] = []
    try:
        raw_manifest = _download(_MANIFEST_URL)
    except Exception:
        local_manifest = _LOCAL_SKILLS_ROOT / "manifest.txt"
        if not local_manifest.exists():
            raise
        raw_manifest = local_manifest.read_text(encoding="utf-8")

    for line in raw_manifest.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)
    return entries


def _download_file(rel_path: str) -> str:
    try:
        return _download(f"{_GITHUB_RAW_BASE}/{rel_path}")
    except Exception:
        local_file = _LOCAL_SKILLS_ROOT / rel_path
        if local_file.exists():
            return local_file.read_text(encoding="utf-8")
        raise


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    shutil.rmtree(path)


def _install_to(target: Path, force: bool) -> Path:
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    dest = target / DEFAULT_SKILL_ID

    if dest.exists():
        if not force:
            raise SystemExit(
                f"Skill already exists at {dest}.\n"
                "Re-run with --force to overwrite."
            )
        _remove_existing(dest)

    for rel_path in _download_manifest():
        content = _download_file(rel_path)
        output_file = dest / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")

    return dest


def add_skill(args: argparse.Namespace) -> None:
    if not (args.claude or args.codex or args.opencode or args.dest):
        print(
            "Pick a destination via --claude, --codex, --opencode, or --dest.",
            file=sys.stderr,
        )
        sys.exit(1)

    targets_dict = GLOBAL_TARGETS if args.global_ else LOCAL_TARGETS
    targets: list[Path] = []

    if args.claude:
        targets.append(targets_dict["claude"])
    if args.codex:
        targets.append(targets_dict["codex"])
    if args.opencode:
        targets.append(targets_dict["opencode"])
    if args.dest:
        targets.append(args.dest)

    for target in targets:
        installed_path = _install_to(target, force=args.force)
        print(f"Installed '{DEFAULT_SKILL_ID}' to {installed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Install kernels skills for AI coding assistants",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Install for Claude.",
    )
    parser.add_argument(
        "--codex",
        action="store_true",
        help="Install for Codex.",
    )
    parser.add_argument(
        "--opencode",
        action="store_true",
        help="Install for OpenCode.",
    )
    parser.add_argument(
        "--global",
        "-g",
        dest="global_",
        action="store_true",
        help="Install globally (user-level) instead of in the current project directory.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Install into a custom destination (path to skills directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skills in the destination.",
    )
    args = parser.parse_args()
    add_skill(args)


if __name__ == "__main__":
    main()
