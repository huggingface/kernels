import shutil
import sys
from argparse import Namespace
from pathlib import Path

from huggingface_hub.utils import get_session

DEFAULT_SKILL_ID = "cuda-kernels"
SUPPORTED_SKILL_IDS = ("cuda-kernels", "rocm-kernels")
_GITHUB_RAW_BASE_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/kernels/main/skills/{skill_id}"
)
_LOCAL_SKILLS_DIR = Path(__file__).resolve().parents[4] / "skills"

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


def _github_raw_base(skill_id: str) -> str:
    return _GITHUB_RAW_BASE_TEMPLATE.format(skill_id=skill_id)


def _manifest_url(skill_id: str) -> str:
    return f"{_github_raw_base(skill_id)}/manifest.txt"


def _local_skill_root(skill_id: str) -> Path:
    return _LOCAL_SKILLS_DIR / skill_id


def _download_manifest(skill_id: str) -> list[str]:
    entries: list[str] = []
    try:
        raw_manifest = _download(_manifest_url(skill_id))
    except Exception:
        local_manifest = _local_skill_root(skill_id) / "manifest.txt"
        if not local_manifest.exists():
            raise
        raw_manifest = local_manifest.read_text(encoding="utf-8")

    for line in raw_manifest.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)
    return entries


def _download_file(skill_id: str, rel_path: str) -> str:
    try:
        return _download(f"{_github_raw_base(skill_id)}/{rel_path}")
    except Exception:
        local_file = _local_skill_root(skill_id) / rel_path
        if local_file.exists():
            return local_file.read_text(encoding="utf-8")
        raise


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return

    shutil.rmtree(path)


def _install_to(target: Path, force: bool, skill_id: str) -> Path:
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    dest = target / skill_id

    if dest.exists():
        if not force:
            raise SystemExit(
                f"Skill already exists at {dest}.\n" "Re-run with --force to overwrite."
            )
        _remove_existing(dest)

    for rel_path in _download_manifest(skill_id):
        content = _download_file(skill_id, rel_path)
        output_file = dest / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(content, encoding="utf-8")

    return dest


def add_skill(args: Namespace) -> None:
    skill_id = getattr(args, "skill", DEFAULT_SKILL_ID)
    if skill_id not in SUPPORTED_SKILL_IDS:
        supported = ", ".join(SUPPORTED_SKILL_IDS)
        raise SystemExit(f"Unsupported skill '{skill_id}'. Supported skills: {supported}")

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
        installed_path = _install_to(target, force=args.force, skill_id=skill_id)
        print(f"Installed '{skill_id}' to {installed_path}")
