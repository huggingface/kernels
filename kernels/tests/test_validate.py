import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from cryptography.x509 import Certificate
from sigstore.errors import VerificationError
from sigstore.verify import policy as sigstore_policy
from sigstore.verify.policy import VerificationPolicy

import kernels
import kernels.validate as validate_module
import kernels.verify as verify_module
import kernels.verify_cache as verify_cache_module
from kernels._data import Metadata, Version
from kernels.deps import DepTreeNode
from kernels.resolver import LocalKernel
from kernels.validate import (
    ArchValidator,
    DirtyValidator,
    KernelValidator,
    MinverValidator,
    SignatureValidator,
    _installed_version,
    default_kernel_validators,
    default_metadata_validators,
)
from kernels.verify import VerificationResult
from kernels.verify_cache import VerificationReceipt, receipt_key, store_receipt

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


@pytest.fixture
def receipt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_cache_module, "_receipt_dir", lambda: tmp_path)
    return tmp_path


class _AcceptPolicy:
    """Test policy that accepts any signer."""

    def verify(self, cert: Certificate) -> None: ...


class _RejectPolicy:
    """Test policy that rejects any signer."""

    def verify(self, cert: Certificate) -> None:
        raise VerificationError("test policy rejects the signer")


def _seed_receipt(kernel: LocalKernel, certificate: Certificate, **fields: str) -> None:
    """Store a receipt under the kernel's key."""
    fields.setdefault("variant", kernel.variant_str)
    store_receipt(
        receipt_key(kernel),
        VerificationReceipt(
            certificate=certificate,
            digest="test-digest",
            verified_at=datetime.now(timezone.utc),
            **fields,
        ),
    )


def test_signature_validator_logs_info_on_success(receipt_dir, monkeypatch, caplog, make_metadata):
    def fake_verify_variant(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        return VerificationResult.Success()

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)
    assert f"Kernel successfully verified: {kernel.variant_path}" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_signature_validator_warns_on_failure(receipt_dir, monkeypatch, caplog, make_metadata):
    # Mixed case on purpose: the reason must be logged as-is and not
    # lowercased (e.g. by str.capitalize()).
    reason = "expected OIDC issuer https://token.actions.githubusercontent.com"

    def fake_verify_variant(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        return VerificationResult.SignatureVerificationFailure(reason=reason)

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)
    assert f"Metadata signature verification failed:\n{reason}: {kernel.variant_path}" in caplog.text


def test_signature_validator_noop_without_sigstore(receipt_dir, monkeypatch, caplog, make_metadata):
    monkeypatch.setattr(validate_module, "has_sigstore", False)

    def fail_if_called(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        raise AssertionError("verify_variant must not be called without sigstore")

    monkeypatch.setattr(verify_module, "verify_variant", fail_if_called)
    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
        SignatureValidator().validate_kernel(kernel=kernel)
    assert not caplog.records


def test_signature_validator_passes_policy_through(receipt_dir, monkeypatch, make_metadata):
    seen_policies: list[VerificationPolicy | None] = []

    def fake_verify_variant(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        seen_policies.append(policy)
        return VerificationResult.Success()

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)

    test_policy = sigstore_policy.Identity(identity="me@danieldk.eu", issuer="https://github.com/login/oauth")
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    for verification_policy in (test_policy, None):
        SignatureValidator(verification_policy).validate_kernel(kernel=kernel)

    assert seen_policies == [test_policy, None]


def test_signature_validator_accepts_valid_receipt(receipt_dir, test_certificate, make_metadata, monkeypatch, caplog):
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    _seed_receipt(kernel, test_certificate, path=str(kernel.variant_path.resolve()))

    def fail_if_called(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        raise AssertionError("verify_variant must not be called on a receipt hit")

    monkeypatch.setattr(verify_module, "verify_variant", fail_if_called)
    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        SignatureValidator(_AcceptPolicy()).validate_kernel(kernel=kernel)

    assert f"Kernel already verified: {kernel.variant_path}" in caplog.text


def test_signature_validator_rejects_signer_from_receipt(
    receipt_dir, test_certificate, make_metadata, monkeypatch, caplog
):
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    _seed_receipt(kernel, test_certificate, path=str(kernel.variant_path.resolve()))

    def fail_if_called(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        raise AssertionError("rejection from a receipt is final, no re-verification")

    monkeypatch.setattr(verify_module, "verify_variant", fail_if_called)
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        SignatureValidator(_RejectPolicy()).validate_kernel(kernel=kernel)

    assert "Metadata signature verification failed" in caplog.text


def test_signature_validator_reverifies_on_receipt_mismatch(
    receipt_dir, test_certificate, make_metadata, monkeypatch, caplog
):
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    # The receipt is stored under this kernel's key, but describes another kernel.
    _seed_receipt(kernel, test_certificate, path="/somewhere/else")

    verified: list[Path] = []

    def fake_verify_variant(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        verified.append(variant_path)
        return VerificationResult.Success()

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)

    assert "Verification receipt does not match kernel" in caplog.text
    assert verified == [kernel.variant_path]


def test_signature_validator_verify_always_bypasses_cache(
    receipt_dir, test_certificate, make_metadata, monkeypatch, caplog
):
    monkeypatch.setenv("KERNELS_VERIFY", "always")
    kernel = LocalKernel(Path("test-variant"), make_metadata("cuda", None))
    _seed_receipt(kernel, test_certificate, path=str(kernel.variant_path.resolve()))

    verified: list[Path] = []

    def fake_verify_variant(variant_path: Path, policy: VerificationPolicy | None = None) -> VerificationResult.Any:
        verified.append(variant_path)
        return VerificationResult.Success()

    monkeypatch.setattr(verify_module, "verify_variant", fake_verify_variant)
    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        SignatureValidator().validate_kernel(kernel=kernel)

    assert verified == [kernel.variant_path]
    assert f"Kernel successfully verified: {kernel.variant_path}" in caplog.text


def test_kernel_validator_checks_entire_dependency_tree(make_metadata):
    tree = DepTreeNode(
        location=LocalKernel(Path("root-variant"), make_metadata("cuda", ["8.0"])),
        deps={
            "test/dependency": DepTreeNode(
                location=LocalKernel(Path("dependency-variant"), make_metadata("cuda", ["9.0"])),
                deps={},
            )
        },
    )
    validated: list[tuple[str, Path]] = []

    class RecordingValidator:
        def validate_kernel(self, *, kernel: LocalKernel):
            validated.append((kernel.variant_str, kernel.variant_path))

    validator: KernelValidator = RecordingValidator()
    tree.validate_kernel(validator)

    assert validated == [
        ("root-variant", Path("root-variant")),
        ("dependency-variant", Path("dependency-variant")),
    ]


def test_default_kernel_validators_include_signature_validator():
    assert any(isinstance(validator, SignatureValidator) for validator in default_kernel_validators())
