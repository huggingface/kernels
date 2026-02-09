import argparse
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

import tomlkit
from kernels.utils import KNOWN_BACKENDS


def parse_kernel_name(value: str) -> NamedTuple:
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):  # validate format
        raise argparse.ArgumentTypeError("must be <owner>/<repo>")
    owner, name = parts

    if "/" in name or "\\" in name:  # validate kernel name
        raise argparse.ArgumentTypeError("repo name cannot contain path separators")

    name = name.lower().replace("-", "_")  # normalize name
    RepoInfo = NamedTuple("RepoInfo", [("name", str), ("owner", str), ("repo_id", str)])
    return RepoInfo(name=name, owner=owner, repo_id=f"{owner}/{name}")


def run_init(args: Namespace) -> None:
    kernel_name = args.kernel_name.name
    repo_id = args.kernel_name.repo_id
    backends = KNOWN_BACKENDS if "all" in args.backends else set(args.backends)

    # Target directory
    target_dir = Path.cwd() / kernel_name

    if args.overwrite:
        if target_dir.exists():
            shutil.rmtree(target_dir)

    if target_dir.exists() and any(target_dir.iterdir()):
        print(
            f"Error: Directory already exists and is not empty: {target_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Suppress progress bars for cleaner output (files are often cached)
    disable_progress_bars()

    print(f"Downloading template from {args.template_repo}...", file=sys.stderr)
    template_dir = Path(
        snapshot_download(repo_id=args.template_repo, repo_type="model")
    )
    _init_from_local_template(template_dir, target_dir, kernel_name, repo_id)

    if backends:
        _update_build_backends(target_dir / "build.toml", backends)

        # replacement logic
        # - rocm uses cuda source so we need to replace the rocm with cuda
        if "rocm" in backends:
            backends.remove("rocm")
            backends.add("cuda")

        _remove_backend_dirs(target_dir, backends)

    # Initialize git repo (required for Nix flakes)
    subprocess.run(["git", "init"], cwd=target_dir, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=target_dir, check=True, capture_output=True)

    print(f"Initialized kernel project: {target_dir}")
    _print_tree(target_dir)
    print("\nNext steps:\n")
    print(f"cd {kernel_name}")
    print("cachix use huggingface")
    print("nix run -L --max-jobs 1 --cores 8 .#build-and-copy")
    print("uv run example.py")


def _init_from_local_template(
    template_dir: Path,
    target_dir: Path,
    kernel_name: str,
    repo_id: str,
) -> None:
    # Placeholder mappings
    replacements = {
        "__KERNEL_NAME__": kernel_name,
        "__KERNEL_NAME_NORMALIZED__": kernel_name,
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


def _update_build_backends(build_toml_path: Path, backends: set[str]) -> None:
    if not build_toml_path.exists():
        return

    with open(build_toml_path, "rb") as f:
        build_contents = tomlkit.parse(f.read())

    # update backends
    if "general" not in build_contents:
        return
    build_contents["general"]["backends"] = list(backends)  # type: ignore[index]

    # update kernel sections
    if "kernel" in build_contents:
        kernel_table = build_contents["kernel"]
        remove_kernels = []
        for name, cfg in kernel_table.items():  # type: ignore[union-attr]
            if isinstance(cfg, dict) and cfg.get("backend") not in set(backends):
                remove_kernels.append(name)
        for name in remove_kernels:
            del kernel_table[name]  # type: ignore[union-attr]

    # write back to file
    with open(build_toml_path, "wb") as f:
        f.write(tomlkit.dumps(build_contents).encode("utf-8"))


def _remove_backend_dirs(target_dir: Path, backends: set[str]) -> None:
    keep = set(backends)
    known = set(KNOWN_BACKENDS)
    for entry in target_dir.iterdir():
        if not entry.is_dir():
            continue
        for backend in known - keep:
            if entry.name.endswith(f"_{backend}"):
                shutil.rmtree(entry)
                break
