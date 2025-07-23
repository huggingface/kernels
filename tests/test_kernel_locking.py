from dataclasses import dataclass
from pathlib import Path

import pytest
import torch.nn as nn

from kernels import load_kernel
from kernels.cli import download_kernels
from kernels.layer import (
    LockedLayerRepository,
    Mode,
    kernelize,
    use_kernel_forward_from_hub,
    use_kernel_mapping,
)


# Mock download arguments class.
@dataclass
class DownloadArgs:
    all_variants: bool
    project_dir: Path


def test_download_all_hash_validation():
    project_dir = Path(__file__).parent / "kernel_locking"
    download_kernels(DownloadArgs(all_variants=True, project_dir=project_dir))


@pytest.mark.linux_only
def test_load_locked():
    project_dir = Path(__file__).parent / "kernel_locking"
    # Also validates that hashing works correctly.
    download_kernels(DownloadArgs(all_variants=False, project_dir=project_dir))
    load_kernel("kernels-community/activation", lockfile=project_dir / "kernels.lock")


def test_layer_locked():
    project_dir = Path(__file__).parent / "layer_locking"

    @use_kernel_forward_from_hub("Version")
    class Version(nn.Module):
        def forward(self) -> str:
            return "0.0.0"

    version = Version()

    with use_kernel_mapping(
        {
            "Version": {
                "cuda": LockedLayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    lockfile=project_dir / "kernels.lock",
                )
            },
        }
    ):
        version = kernelize(version, device="cuda", mode=Mode.INFERENCE)
        assert version() == "0.1.1"
