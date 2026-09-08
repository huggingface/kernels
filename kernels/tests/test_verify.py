import logging
from pathlib import Path
from types import ModuleType

import pytest
from sigstore.verify import policy

import kernels.verify as verify_module
import kernels.verify_cache as verify_cache_module
from kernels import install_kernel
from kernels._data import DigestViolation, KernelDependency, KernelVersion
from kernels._versions import select_revision_or_version
from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.load import get_kernel_with_resolver
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS, HubResolver
from kernels.validate import AllKernelValidator, AllValidator, SignatureValidator
from kernels.verify import VerificationResult, verify_variant

TEST_POLICY: policy.VerificationPolicy = policy.Identity(
    identity="me@danieldk.eu", issuer="https://github.com/login/oauth"
)


@pytest.fixture
def receipt_dir(tmp_path, monkeypatch):
    """Isolate the verification receipt cache from the real kernel cache."""
    monkeypatch.setattr(verify_cache_module, "_receipt_dir", lambda: tmp_path)
    return tmp_path


def test_correctly_signed_kernel_passes_with_default_policy():
    revision = select_revision_or_version("kernels-community/relu", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-community/relu", revision=revision)
    assert verify_variant(variant_path) == VerificationResult.Success()


def test_correctly_signed_kernel_passes():
    revision = select_revision_or_version("kernels-test/signatures", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-test/signatures", revision=revision)
    assert (
        verify_variant(
            variant_path,
            policy=TEST_POLICY,
        )
        == VerificationResult.Success()
    )


def _load_signatures_kernel(version: KernelVersion) -> ModuleType:
    """Load kernels-test/signatures end-to-end."""
    return get_kernel_with_resolver(
        api=_get_hf_api(),
        backend=None,
        kernel=KernelDependency(repo_id="kernels-test/signatures", version=version),
        resolver=HubResolver(trust_remote_code=True),
        kernel_validator=AllKernelValidator(validators=[SignatureValidator(TEST_POLICY)]),
        metadata_validator=AllValidator(validators=[]),
    )


def test_load_verifies_signature_e2e(receipt_dir, caplog):
    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        _load_signatures_kernel(KernelVersion.Version(1))

    assert "Kernel successfully verified" in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    ("revision", "message"),
    [
        ("signature-invalid", "Metadata signature verification failed"),
        ("signature-missing", "Cannot verify kernel integrity, signature not found"),
    ],
)
def test_load_warns_on_signature_failure_e2e(receipt_dir, caplog, revision: str, message: str):
    with caplog.at_level(logging.WARNING, logger="kernels.validate"):
        # Signature verification failures warn, but do not prevent loading.
        _load_signatures_kernel(KernelVersion.Revision(revision))

    assert message in caplog.text


def test_load_uses_verification_cache_e2e(receipt_dir, monkeypatch, caplog):
    verify_calls: list[Path] = []
    real_verify_variant = verify_module.verify_variant

    def spy_verify_variant(
        variant_path: Path, policy: policy.VerificationPolicy | None = None
    ) -> VerificationResult.Any:
        verify_calls.append(variant_path)
        return real_verify_variant(variant_path, policy=policy)

    monkeypatch.setattr(verify_module, "verify_variant", spy_verify_variant)

    with caplog.at_level(logging.INFO, logger="kernels.validate"):
        _load_signatures_kernel(KernelVersion.Version(1))
        _load_signatures_kernel(KernelVersion.Version(1))

    assert "Kernel successfully verified" in caplog.text
    assert "Kernel already verified" in caplog.text
    assert len(verify_calls) == 1


def test_invalid_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="invalid-digest")

    match verify_variant(
        variant_path,
        policy=TEST_POLICY,
    ):
        case VerificationResult.DigestVerificationFailure(violations=violations):
            assert len(violations) == 1
            assert isinstance(violations[0], DigestViolation.HashMismatch)
        case other:
            raise RuntimeError(f"Expected DigestVerificationFailure, was: {other}")


def test_invalid_metadata_fails():
    # We cannot use regular code paths, because they require valid metadata.
    revision = select_revision_or_version(
        "kernels-test/signatures",
        revision="invalid-metadata",
        version=None,
        local_files_only=False,
    )

    api = _get_hf_api()
    variant_paths = (
        Path(
            str(
                api.snapshot_download(
                    "kernels-test/signatures",
                    repo_type="kernel",
                    allow_patterns="build/*",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=revision,
                )
            )
        )
        / "build"
    )

    match verify_variant(
        # No CUDA dependency, we are only checking metadata.
        variant_paths / "torch-cuda",
        policy=TEST_POLICY,
    ):
        case VerificationResult.MetadataInvalid(reason=reason):
            assert "Cannot parse metadata" in reason
        case other:
            raise RuntimeError(f"Expected MetadataInvalid, was: {other}")


def test_missing_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="missing-digest")
    assert (
        verify_variant(
            variant_path,
            policy=TEST_POLICY,
        )
        == VerificationResult.DigestMissing()
    )


def test_missing_metadata_fails():
    # We cannot use regular code paths, because they require valid metadata.
    revision = select_revision_or_version(
        "kernels-test/signatures",
        revision="missing-metadata",
        version=None,
        local_files_only=False,
    )

    api = _get_hf_api()
    variant_paths = (
        Path(
            str(
                api.snapshot_download(
                    "kernels-test/signatures",
                    repo_type="kernel",
                    allow_patterns="build/*",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=revision,
                )
            )
        )
        / "build"
    )

    assert (
        verify_variant(
            # No CUDA dependency, we are only checking metadata.
            variant_paths / "torch-cuda",
            policy=TEST_POLICY,
        )
        == VerificationResult.MetadataMissing()
    )


def test_unsigned_kernel_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-missing")
    assert (
        verify_variant(
            variant_path,
            policy=TEST_POLICY,
        )
        == VerificationResult.SignatureBundleMissing()
    )


def test_broken_signature_bundle_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-broken")
    match verify_variant(
        variant_path,
        policy=TEST_POLICY,
    ):
        case VerificationResult.SignatureBundleInvalid(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureBundleInvalid, was: {other}")


def test_invalid_signature_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-invalid")
    match verify_variant(
        variant_path,
        policy=TEST_POLICY,
    ):
        case VerificationResult.SignatureVerificationFailure(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")
