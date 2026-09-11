import json
import logging
import re
from pathlib import Path

import pytest
import torch

import kernels
import kernels.validate as validate_module
import kernels.verify as verify_module
from kernels._rust import KernelLocation, Metadata, Version
from kernels.deps import DepTreeNode
from kernels.resolver import LocalKernel, RemoteKernel
from kernels.validate import (
    ArchValidator,
    DirtyValidator,
    MinverValidator,
    SignatureValidator,
    _installed_version,
    default_metadata_validators,
)
from kernels.variants import parse_variant
from kernels.verify import VerificationResult

CLEAN_PROVENANCE = {
    "kernel-builder": {"version": "0.1.0", "commit": "a" * 40, "dirty": False},
    "kernel": {"commit": "b" * 40, "dirty": False},
}
DIRTY_KERNEL = {
    "kernel-builder": {"version": "0.1.0", "commit": "a" * 40, "dirty": False},
    "kernel": {"commit": "b" * 40, "dirty": True},
}
DIRTY_BUILDER = {
    "kernel-builder": {"version": "0.1.0", "commit": "a" * 40, "dirty": True},
    "kernel": {"commit": "b" * 40, "dirty": False},
}


def _metadata_with_provenance(provenance):
    metadata = {
        "id": "activation_1_cuda",
        "name": "activation",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": {"type": "cuda"},
    }
    if provenance is not None:
        metadata["provenance"] = provenance
    return Metadata.from_bytes(json.dumps(metadata).encode())


@pytest.mark.parametrize("provenance", [None, CLEAN_PROVENANCE])
def test_dirty_validator_does_not_warn_when_clean(caplog, provenance):
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        DirtyValidator().validate_metadata(metadata=_metadata_with_provenance(provenance), variant="test-variant")
    assert not caplog.records


def test_dirty_validator_warns_on_dirty_kernel_source(caplog):
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        DirtyValidator().validate_metadata(metadata=_metadata_with_provenance(DIRTY_KERNEL), variant="test-variant")
    assert "dirty git tree" in caplog.text


def test_dirty_validator_names_dirty_sources(caplog):
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        DirtyValidator().validate_metadata(metadata=_metadata_with_provenance(DIRTY_BUILDER), variant="test-variant")
    assert "kernel-builder" in caplog.text


def test_dirty_validator_is_enabled_by_default():
    assert any(isinstance(validator, DirtyValidator) for validator in default_metadata_validators())


@pytest.mark.parametrize("minver", [None, "0.0.1"])
def test_no_error_when_minver_met(minver, make_metadata):
    MinverValidator().validate_metadata(
        metadata=make_metadata("cuda", None, kernels_minver=minver), variant="test-variant"
    )


def test_no_error_for_dev_version_of_required_release(monkeypatch, make_metadata):
    # A development version implements the release it leads up to, so
    # `0.17.0.dev0` must satisfy a `0.17.0` requirement.
    monkeypatch.setattr(kernels, "__version__", "0.17.0.dev0")
    MinverValidator().validate_metadata(
        metadata=make_metadata("cuda", None, kernels_minver="0.17.0"), variant="test-variant"
    )


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


def test_unparseable_installed_version_does_not_fail_validation(monkeypatch, make_metadata):
    # A version that cannot be compared must not turn this check into a
    # failure.
    monkeypatch.setattr(kernels, "__version__", "0.17.0-dirty")
    MinverValidator().validate_metadata(
        metadata=make_metadata("cuda", None, kernels_minver="999.1.0"), variant="test-variant"
    )


def test_version_ordering_is_numeric_not_lexicographic():
    # `0.9 < 0.10` only holds for numeric comparison; string comparison would
    # get this backwards.
    assert Version.from_str("0.9") < Version.from_str("0.10")
    assert Version.from_str("0.14") == Version.from_str("0.14.0")
    assert Version.from_str("0.14.0") < Version.from_str("0.14.1")


def test_raises_when_minver_not_met(make_metadata):
    with pytest.raises(RuntimeError, match="requires kernels>=999.1"):
        MinverValidator().validate_metadata(
            metadata=make_metadata("cuda", None, kernels_minver="999.1.0"), variant="test-variant"
        )


def test_error_mentions_installed_version(make_metadata):
    with pytest.raises(RuntimeError, match=f"version {re.escape(kernels.__version__)} is installed"):
        MinverValidator().validate_metadata(
            metadata=make_metadata("cuda", None, kernels_minver="999.1.0"), variant="test-variant"
        )


def test_cuda_incompatible_arch_is_rejected(fake_cuda_device, make_metadata):
    with pytest.raises(RuntimeError) as exc_info:
        ArchValidator().validate_metadata(metadata=make_metadata("cuda", ["8.0", "9.0a"]), variant="test-variant")
    assert "test-variant" in str(exc_info.value)
    assert "CUDA capability 10.0" in str(exc_info.value)
    assert "8.0, 9.0a" in str(exc_info.value)


def test_cuda_compatible_arch_is_accepted(fake_cuda_device, make_metadata):
    for archs in (["8.0", "10.0"], ["10.0a"], ["10.0f"]):
        ArchValidator().validate_metadata(metadata=make_metadata("cuda", archs), variant="test-variant")


def test_noarch_build_is_accepted(fake_cuda_device, make_metadata):
    # Backends that support archs (e.g. CUDA) can have builds that do not
    # declare any (noarch kernels, e.g. pure Triton builds). Such builds are
    # never rejected.
    ArchValidator().validate_metadata(metadata=make_metadata("cuda", None), variant="test-variant")
    ArchValidator().validate_metadata(metadata=make_metadata("cuda", []), variant="test-variant")


def test_rocm_arch_check(fake_rocm_device, make_metadata):
    ArchValidator().validate_metadata(metadata=make_metadata("rocm", ["gfx90a", "gfx942"]), variant="test-variant")
    ArchValidator().validate_metadata(metadata=make_metadata("rocm", None), variant="test-variant")
    with pytest.raises(RuntimeError) as exc_info:
        ArchValidator().validate_metadata(metadata=make_metadata("rocm", ["gfx942"]), variant="test-variant")
    assert "ROCm arch gfx90a" in str(exc_info.value)
    assert "gfx942" in str(exc_info.value)


def test_check_skipped_without_device(monkeypatch, make_metadata):
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ArchValidator().validate_metadata(metadata=make_metadata("cuda", ["8.0"]), variant="test-variant")


def test_check_skipped_for_backends_without_archs(fake_cuda_device, make_metadata):
    # Archs of other backends cannot be checked against the current device.
    ArchValidator().validate_metadata(metadata=make_metadata("cpu", None), variant="test-variant")
    ArchValidator().validate_metadata(metadata=make_metadata("metal", ["applegpu_g13"]), variant="test-variant")


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


def test_issue_707_fa3_on_b200(fake_cuda_device, make_metadata):
    # https://github.com/huggingface/kernels/issues/707: flash-attn3 only
    # declares sm_80/sm_90a archs, but loading it on a B200 (capability 10.0)
    # succeeded and the first launch exited the process. The declared archs
    # must be rejected for this device.
    metadata = make_metadata("cuda", ["8.0", "9.0a"])
    with pytest.raises(RuntimeError, match="does not support the current device"):
        ArchValidator().validate_metadata(metadata=metadata, variant="test-variant")


_SIGNED_REPO_ID = "kernels-test/signatures"
_SIGNED_REVISION = "a" * 40


def _hub_kernel(tmp_path, metadata) -> LocalKernel:
    variant_path = tmp_path / "torch-cuda"
    return LocalKernel(
        variant_path=variant_path,
        metadata=metadata,
        origin=RemoteKernel(
            repo_id=_SIGNED_REPO_ID,
            revision=_SIGNED_REVISION,
            metadata=metadata,
            variant=parse_variant("torch-cuda"),
        ),
    )


@pytest.fixture
def recorded_verifications(monkeypatch):
    """Record `verify_variant` calls and control the result it returns."""
    calls = []
    results = []

    def fake_verify_variant(variant_path, *, location, policy=None, cache=True):
        calls.append({"variant_path": variant_path, "policy": policy, "location": location, "cache": cache})
        return results.pop(0) if results else VerificationResult.Success()

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)
    return calls, results


def test_signature_validator_skips_local_kernels(tmp_path, make_metadata, recorded_verifications):
    calls, _ = recorded_verifications
    kernel = LocalKernel(variant_path=tmp_path / "torch-cuda", metadata=make_metadata("cuda", None))

    SignatureValidator().validate_kernel(kernel=kernel)

    assert calls == []


def test_signature_validator_identifies_kernel_by_origin(tmp_path, make_metadata, recorded_verifications):
    calls, _ = recorded_verifications
    kernel = _hub_kernel(tmp_path, make_metadata("cuda", None))

    SignatureValidator().validate_kernel(kernel=kernel)

    (call,) = calls
    assert call["variant_path"] == kernel.variant_path
    assert call["location"] == KernelLocation.remote(_SIGNED_REPO_ID, _SIGNED_REVISION, "torch-cuda")
    # Loading a kernel must reuse a previous verification.
    assert call["cache"] is True


def test_signature_validator_passes_policy(tmp_path, make_metadata, recorded_verifications):
    calls, _ = recorded_verifications
    kernel = _hub_kernel(tmp_path, make_metadata("cuda", None))
    sentinel = object()

    SignatureValidator(policy=sentinel).validate_kernel(kernel=kernel)

    (call,) = calls
    assert call["policy"] is sentinel


def test_signature_validator_is_quiet_on_success(tmp_path, make_metadata, recorded_verifications, caplog):
    kernel = _hub_kernel(tmp_path, make_metadata("cuda", None))

    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)

    assert caplog.text == ""


@pytest.mark.parametrize(
    "result",
    [
        VerificationResult.SignatureBundleMissing(),
        VerificationResult.SignatureBundleInvalid(reason="bad bundle"),
        VerificationResult.SignatureVerificationFailure(reason="bad signature"),
        VerificationResult.MetadataInvalid(reason="bad metadata"),
        VerificationResult.MetadataMissing(),
        VerificationResult.DigestMissing(),
        VerificationResult.DigestVerificationFailure(violations=[]),
    ],
)
def test_signature_validator_warns_but_does_not_raise(tmp_path, make_metadata, recorded_verifications, caplog, result):
    _, results = recorded_verifications
    results.append(result)
    kernel = _hub_kernel(tmp_path, make_metadata("cuda", None))

    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)

    # The message belongs to the result. The validator only says which kernel
    # it applies to, so the wording is asserted where it is defined.
    assert str(result) in caplog.text
    assert "test-kernel" in caplog.text
