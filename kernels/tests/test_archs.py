import json

import pytest
import torch
from kernels_data import Metadata

from kernels import get_kernel, has_kernel, install_kernel
from kernels.archs import _check_arch_incompatibility, _supports_cuda_capability


def make_metadata(backend_type: str, archs: list[str] | None) -> Metadata:
    return Metadata.from_bytes(
        json.dumps(
            {
                "name": "test-kernel",
                "id": "test_id",
                "version": 1,
                "license": "mit",
                "python-depends": [],
                "backend": {"type": backend_type, "archs": archs},
            }
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    "archs,capability,supported",
    [
        # Base archs run on the same generation with the same or a newer
        # minor capability.
        (["8.0"], (8, 0), True),
        (["8.0"], (8, 6), True),
        (["8.6"], (8, 0), False),
        (["8.0"], (7, 5), False),
        (["8.0"], (9, 0), False),
        (["8.0", "9.0"], (9, 0), True),
        # Architecture-specific archs only run on that exact capability.
        (["9.0a"], (9, 0), True),
        (["9.0a"], (9, 1), False),
        (["8.0", "9.0a"], (10, 0), False),
        # Family-specific archs run on the same generation with the same or
        # a newer minor capability.
        (["10.0f"], (10, 0), True),
        (["10.0f"], (10, 3), True),
        (["10.0f"], (12, 0), False),
        # PTX is JIT-compiled for the current device, so builds with `+PTX`
        # also run on any newer capability.
        (["9.0+PTX"], (9, 0), True),
        (["9.0+PTX"], (9, 1), True),
        (["9.0+PTX"], (10, 0), True),
        (["9.0+PTX"], (12, 1), True),
        (["9.0+PTX"], (8, 6), False),
        # ...but the `a` and `f` suffixes keep their own semantics.
        (["9.0a+PTX"], (9, 0), True),
        (["9.0a+PTX"], (10, 0), False),
        (["10.0f+PTX"], (10, 3), True),
        (["10.0f+PTX"], (12, 0), False),
        # Arch strings in an unknown format do not count as a match...
        (["garbage", "8.0"], (9, 0), False),
        # ...but when no arch string can be parsed, compatibility cannot be
        # determined and the build is not rejected.
        (["garbage"], (9, 0), True),
    ],
)
def test_supports_cuda_capability(archs, capability, supported):
    assert _supports_cuda_capability(archs, capability) == supported


@pytest.fixture
def fake_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))


@pytest.fixture
def fake_rocm_device(monkeypatch):
    class FakeProperties:
        gcnArchName = "gfx90a:sramecc+:xnack-"

    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: FakeProperties())


def test_cuda_incompatible_arch_is_rejected(fake_cuda_device):
    with pytest.raises(RuntimeError) as exc_info:
        _check_arch_incompatibility(make_metadata("cuda", ["8.0", "9.0a"]), "test-variant")
    assert "test-variant" in str(exc_info.value)
    assert "CUDA capability 10.0" in str(exc_info.value)
    assert "8.0, 9.0a" in str(exc_info.value)


def test_cuda_compatible_arch_is_accepted(fake_cuda_device):
    for archs in (["8.0", "10.0"], ["10.0a"], ["10.0f"], None):
        _check_arch_incompatibility(make_metadata("cuda", archs), "test-variant")


def test_rocm_arch_check(fake_rocm_device):
    _check_arch_incompatibility(make_metadata("rocm", ["gfx90a", "gfx942"]), "test-variant")
    _check_arch_incompatibility(make_metadata("rocm", None), "test-variant")
    with pytest.raises(RuntimeError) as exc_info:
        _check_arch_incompatibility(make_metadata("rocm", ["gfx942"]), "test-variant")
    assert "ROCm arch gfx90a" in str(exc_info.value)
    assert "gfx942" in str(exc_info.value)


def test_check_skipped_without_device(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _check_arch_incompatibility(make_metadata("cuda", ["8.0"]), "test-variant")


def test_check_skipped_for_backends_without_archs(fake_cuda_device):
    # Archs of other backends cannot be checked against the current device.
    _check_arch_incompatibility(make_metadata("cpu", None), "test-variant")
    _check_arch_incompatibility(make_metadata("metal", ["applegpu_g13"]), "test-variant")


def test_issue_707_fa3_on_b200(fake_cuda_device):
    # https://github.com/huggingface/kernels/issues/707: flash-attn3 only
    # declares sm_80/sm_90a archs, but loading it on a B200 (capability 10.0)
    # succeeded and the first launch exited the process. The declared archs
    # must be rejected for this device.
    metadata = make_metadata("cuda", ["8.0", "9.0a"])
    with pytest.raises(RuntimeError, match="does not support the current device"):
        _check_arch_incompatibility(metadata, "test-variant")


@pytest.mark.cuda_only
def test_get_kernel_rejects_unsupported_capability(monkeypatch):
    variant_path = install_kernel("kernels-community/relu", revision="v1")
    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    if not metadata.backend.archs:
        pytest.skip("kernel build does not declare archs")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (99, 9))

    with pytest.raises(RuntimeError, match="does not support the current device"):
        get_kernel("kernels-community/relu", version=1)
    assert not has_kernel("kernels-community/relu", version=1)

    # The opt-out only checks that a compatible build variant exists.
    assert has_kernel("kernels-community/relu", version=1, check_arch=False)
    kernel = get_kernel("kernels-community/relu", version=1, check_arch=False)
    assert kernel is not None
