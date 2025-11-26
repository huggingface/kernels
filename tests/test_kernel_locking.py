from dataclasses import dataclass
from pathlib import Path

import pytest
import torch.nn as nn

from kernels import (
    LockedFuncRepository,
    LockedLayerRepository,
    Mode,
    kernelize,
    load_kernel,
    use_kernel_forward_from_hub,
    use_kernel_func_from_hub,
    use_kernel_mapping,
)
from kernels.cli import download_kernels


# Mock download arguments class.
@dataclass
class DownloadArgs:
    all_variants: bool
    project_dir: Path


def test_download_all_hash_validation():
    project_dir = Path(__file__).parent / "kernel_locking"
    download_kernels(DownloadArgs(all_variants=True, project_dir=project_dir))


@pytest.mark.cuda_only
def test_load_locked():
    project_dir = Path(__file__).parent / "kernel_locking"
    # Also validates that hashing works correctly.
    download_kernels(DownloadArgs(all_variants=False, project_dir=project_dir))
    load_kernel("kernels-community/activation", lockfile=project_dir / "kernels.lock")


def test_layer_locked(device):
    project_dir = Path(__file__).parent / "layer_locking"

    @use_kernel_forward_from_hub("Version")
    class Version(nn.Module):
        def forward(self) -> str:
            return "0.0.0"

    version = Version()

    with use_kernel_mapping(
        {
            "Version": {
                device: LockedLayerRepository(
                    repo_id="kernels-test/versions",
                    layer_name="Version",
                    lockfile=project_dir / "kernels.lock",
                )
            },
        }
    ):
        version = kernelize(version, device=device, mode=Mode.INFERENCE)
        assert version() == "0.1.1"


def test_func_locked(device):
    project_dir = Path(__file__).parent / "layer_locking"

    @use_kernel_func_from_hub("version")
    def version():
        return "0.0.0"

    class Version(nn.Module):
        def __init__(self):
            super().__init__()
            self.version = version

        def forward(self) -> str:
            return self.version()

    model = Version()

    print(model.version.forward)

    with use_kernel_mapping(
        {
            "version": {
                device: LockedFuncRepository(
                    repo_id="kernels-test/versions",
                    func_name="version",
                    lockfile=project_dir / "kernels.lock",
                )
            },
        }
    ):
        model = kernelize(model, device=device, mode=Mode.INFERENCE)

    assert version() == "0.1.1"

    print(model.version.forward)

    with use_kernel_mapping({"version": {}}):
        model = kernelize(model, mode=Mode.INFERENCE, device=device)

    assert version() == "0.0.0"

    print(model.version.forward)
