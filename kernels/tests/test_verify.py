from pathlib import Path

from kernels_data import DigestViolation
from sigstore.verify import policy

from kernels import install_kernel
from kernels._versions import select_revision_or_version
from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS
from kernels.verify import VerificationResult, verify_variant

TEST_POLICY: policy.VerificationPolicy = policy.Identity(
    identity="me@danieldk.eu", issuer="https://github.com/login/oauth"
)


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
