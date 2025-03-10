from dataclasses import dataclass
from pathlib import Path

from kernels import load_kernel
from kernels.cli import download_kernels


# Mock download arguments class.
@dataclass
class DownloadArgs:
    all_variants: bool
    project_dir: Path


def test_download_all_hash_validation():
    project_dir = Path(__file__).parent / "kernel_locking"
    download_kernels(DownloadArgs(all_variants=True, project_dir=project_dir))


def test_load_locked():
    project_dir = Path(__file__).parent / "kernel_locking"
    # Also validates that hashing works correctly.
    download_kernels(DownloadArgs(all_variants=False, project_dir=project_dir))
    load_kernel("kernels-community/activation", lockfile=project_dir / "kernels.lock")
