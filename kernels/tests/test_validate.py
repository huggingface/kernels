import re
from pathlib import Path

import pytest
import torch
from kernels_data import Version

import kernels
import kernels.validate as validate_module
from kernels.deps import DepTreeNode
from kernels.resolver import LocalKernel
from kernels.validate import ArchValidator, MinverValidator, _installed_version


@pytest.mark.parametrize("minver", [None, "0.0.1"])
def test_no_error_when_minver_met(minver, make_metadata, make_tree):
    MinverValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None, kernels_minver=minver)))


def test_no_error_for_dev_version_of_required_release(monkeypatch, make_metadata, make_tree):
    # A development version implements the release it leads up to, so
    # `0.17.0.dev0` must satisfy a `0.17.0` requirement.
    monkeypatch.setattr(kernels, "__version__", "0.17.0.dev0")
    MinverValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None, kernels_minver="0.17.0")))


@pytest.mark.parametrize(
    "installed",
    ["0.17.0.dev0", "0.17.0rc1", "0.17.0.post1", "0.17.0+cu121", "0.17"],
)
def test_installed_version_uses_release_segment(monkeypatch, installed):
    monkeypatch.setattr(kernels, "__version__", installed)
    assert _installed_version() == Version.from_str("0.17.0")


def test_installed_version_is_none_for_non_pep440_version(monkeypatch):
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    assert _installed_version() is None


def test_unparseable_installed_version_does_not_fail_validation(monkeypatch, make_metadata, make_tree):
    # A version that cannot be compared must not turn this check into a
    # failure.
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    MinverValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None, kernels_minver="999.1.0")))


def test_version_ordering_is_numeric_not_lexicographic():
    # `0.9 < 0.10` only holds for numeric comparison; string comparison would
    # get this backwards.
    assert Version.from_str("0.9") < Version.from_str("0.10")
    assert Version.from_str("0.14") == Version.from_str("0.14.0")
    assert Version.from_str("0.14.0") < Version.from_str("0.14.1")


def test_raises_when_minver_not_met(make_metadata, make_tree):
    with pytest.raises(RuntimeError, match="requires kernels>=999.1"):
        MinverValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None, kernels_minver="999.1.0")))


def test_error_mentions_installed_version(make_metadata, make_tree):
    with pytest.raises(RuntimeError, match=f"version {re.escape(kernels.__version__)} is installed"):
        MinverValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None, kernels_minver="999.1.0")))


def test_cuda_incompatible_arch_is_rejected(fake_cuda_device, make_metadata, make_tree):
    with pytest.raises(RuntimeError) as exc_info:
        ArchValidator().validate_metadata(tree=make_tree(make_metadata("cuda", ["8.0", "9.0a"])))
    assert "test-variant" in str(exc_info.value)
    assert "CUDA capability 10.0" in str(exc_info.value)
    assert "8.0, 9.0a" in str(exc_info.value)


def test_cuda_compatible_arch_is_accepted(fake_cuda_device, make_metadata, make_tree):
    for archs in (["8.0", "10.0"], ["10.0a"], ["10.0f"]):
        ArchValidator().validate_metadata(tree=make_tree(make_metadata("cuda", archs)))


def test_noarch_build_is_accepted(fake_cuda_device, make_metadata, make_tree):
    # Backends that support archs (e.g. CUDA) can have builds that do not
    # declare any (noarch kernels, e.g. pure Triton builds). Such builds are
    # never rejected.
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("cuda", None)))
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("cuda", [])))


def test_rocm_arch_check(fake_rocm_device, make_metadata, make_tree):
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("rocm", ["gfx90a", "gfx942"])))
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("rocm", None)))
    with pytest.raises(RuntimeError) as exc_info:
        ArchValidator().validate_metadata(tree=make_tree(make_metadata("rocm", ["gfx942"])))
    assert "ROCm arch gfx90a" in str(exc_info.value)
    assert "gfx942" in str(exc_info.value)


def test_check_skipped_without_device(monkeypatch, make_metadata, make_tree):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("cuda", ["8.0"])))


def test_check_skipped_for_backends_without_archs(fake_cuda_device, make_metadata, make_tree):
    # Archs of other backends cannot be checked against the current device.
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("cpu", None)))
    ArchValidator().validate_metadata(tree=make_tree(make_metadata("metal", ["applegpu_g13"])))


def test_arch_validator_checks_entire_dependency_tree(monkeypatch, make_metadata):
    tree = DepTreeNode(
        location=LocalKernel(Path("root-variant"), make_metadata("cuda", ["8.0"])),
        deps={
            "test/dependency": DepTreeNode(
                location=LocalKernel(Path("dependency-variant"), make_metadata("cuda", ["9.0"])),
                deps={},
            )
        },
    )
    validated = []

    monkeypatch.setattr(
        validate_module,
        "_check_arch_incompatibility",
        lambda metadata, variant: validated.append((metadata.backend.archs, variant)),
    )

    tree.validate_metadata(ArchValidator())

    assert validated == [
        (["8.0"], "root-variant"),
        (["9.0"], "dependency-variant"),
    ]


def test_issue_707_fa3_on_b200(fake_cuda_device, make_metadata, make_tree):
    # https://github.com/huggingface/kernels/issues/707: flash-attn3 only
    # declares sm_80/sm_90a archs, but loading it on a B200 (capability 10.0)
    # succeeded and the first launch exited the process. The declared archs
    # must be rejected for this device.
    metadata = make_metadata("cuda", ["8.0", "9.0a"])
    with pytest.raises(RuntimeError, match="does not support the current device"):
        ArchValidator().validate_metadata(tree=make_tree(metadata))
