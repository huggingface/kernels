from dataclasses import dataclass
from pathlib import Path

from kernels.cli import download_kernels


# Mock download arguments class.
@dataclass
class DownloadArgs:
    all_variants: bool
    project_dir: Path


def test_download_hash_validation():
    project_dir = Path(__file__).parent / "hash_validation"
    download_kernels(DownloadArgs(all_variants=False, project_dir=project_dir))


def test_download_all_hash_validation():
    project_dir = Path(__file__).parent / "hash_validation"
    download_kernels(DownloadArgs(all_variants=True, project_dir=project_dir))
