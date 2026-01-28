import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars


def run_init(args: Namespace) -> None:
    kernel_name = args.kernel_name
    # must be fully qualified repo name <owner>/<repo>
    owner_repo = kernel_name.split("/")
    if len(owner_repo) != 2:
        print(
            f"Error: kernel_name must be in the format <owner>/<repo> (e.g., drbh/my-kernel)",
            file=sys.stderr,
        )
        sys.exit(1)
    owner, kernel_name = owner_repo
    kernel_name_normalized = kernel_name.replace("-", "_")
    repo_id = f"{owner}/{kernel_name}"
    output_dir = Path.cwd()

    # Validate kernel name
    if "/" in kernel_name or "\\" in kernel_name:
        print(
            f"Error: Kernel name cannot contain path separators: {kernel_name}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Target directory
    target_dir = output_dir / kernel_name
    if target_dir.exists() and any(target_dir.iterdir()):
        print(
            f"Error: Directory already exists and is not empty: {target_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Download template from HuggingFace
    template_repo = args.template_repo

    # Suppress progress bars for cleaner output (files are often cached)
    disable_progress_bars()

    print(f"Downloading template from {template_repo}...", file=sys.stderr)
    template_dir = Path(snapshot_download(repo_id=template_repo, repo_type="model"))
    _init_from_local_template(
        template_dir, target_dir, kernel_name, kernel_name_normalized, repo_id
    )

    # Initialize git repo (required for Nix flakes)
    subprocess.run(["git", "init"], cwd=target_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=target_dir, check=True, capture_output=True)

    print(f"Initialized kernel project: {target_dir}")
    _print_tree(target_dir)
    print("\nNext steps:")
    print(f"  cd {kernel_name}")
    print("  cachix use huggingface")
    print("  nix run -L --max-jobs 1 --cores 8 .#build-and-copy")
    print("  uv run example.py")


def _init_from_local_template(
    template_dir: Path,
    target_dir: Path,
    kernel_name: str,
    kernel_name_normalized: str,
    repo_id: str,
) -> None:
    # Placeholder mappings
    replacements = {
        "__KERNEL_NAME__": kernel_name,
        "__KERNEL_NAME_NORMALIZED__": kernel_name_normalized,
        "__REPO_ID__": repo_id,
    }

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Walk template directory
    for root, dirs, files in os.walk(template_dir):
        # Skip .git directory
        if ".git" in dirs:
            dirs.remove(".git")

        rel_root = Path(root).relative_to(template_dir)

        # Compute target root with placeholder replacement in path
        target_root_str = str(rel_root)
        for old, new in replacements.items():
            target_root_str = target_root_str.replace(old, new)
        target_root = target_dir / target_root_str

        # Create directories
        for dir_name in dirs:
            new_dir_name = dir_name
            for old, new in replacements.items():
                new_dir_name = new_dir_name.replace(old, new)
            (target_root / new_dir_name).mkdir(parents=True, exist_ok=True)

        # Copy and process files
        for file_name in files:

            src_path = Path(root) / file_name

            # Replace placeholders in filename
            new_file_name = file_name
            for old, new in replacements.items():
                new_file_name = new_file_name.replace(old, new)

            dst_path = target_root / new_file_name

            # Read, replace placeholders, write
            try:
                content = src_path.read_text()
                for old, new in replacements.items():
                    content = content.replace(old, new)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                dst_path.write_text(content)
            except UnicodeDecodeError:
                # Binary file, just copy
                shutil.copy2(src_path, dst_path)


def _print_tree(directory: Path, prefix: str = "") -> None:
    entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    entries = [e for e in entries if e.name != ".git"]

    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            _print_tree(entry, prefix + extension)
