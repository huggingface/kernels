from kernels_data import DigestViolation
from sigstore.verify import policy

from kernels import install_kernel
from kernels.utils import select_revision_or_version
from kernels.verify import VerificationResult, verify_variant

TEST_POLICIES: list[policy.VerificationPolicy] = [
    policy.Identity(identity="me@danieldk.eu", issuer="https://github.com/login/oauth")
]


def test_correctly_signed_kernel_passes_with_default_policy():
    revision = select_revision_or_version("kernels-community/relu", revision=None, version=1)
    variant_path = install_kernel("kernels-community/relu", revision=revision)
    assert verify_variant(variant_path) == VerificationResult.Success()


def test_correctly_signed_kernel_passes():
    revision = select_revision_or_version("kernels-test/signatures", revision=None, version=1)
    variant_path = install_kernel("kernels-test/signatures", revision=revision)
    assert (
        verify_variant(
            variant_path,
            policies=TEST_POLICIES,
        )
        == VerificationResult.Success()
    )


def test_invalid_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="invalid-digest")

    match verify_variant(
        variant_path,
        policies=TEST_POLICIES,
    ):
        case VerificationResult.DigestVerificationFailure(violations=violations):
            assert len(violations) == 1
            assert isinstance(violations[0], DigestViolation.HashMismatch)
        case other:
            raise RuntimeError(f"Expected DigestVerificationFailure, was: {other}")


def test_invalid_metadata_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="invalid-metadata")

    match verify_variant(
        variant_path,
        policies=TEST_POLICIES,
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
            policies=TEST_POLICIES,
        )
        == VerificationResult.DigestMissing()
    )


def test_missing_metadata_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="missing-metadata")
    assert (
        verify_variant(
            variant_path,
            policies=TEST_POLICIES,
        )
        == VerificationResult.MetadataMissing()
    )


def test_unsigned_kernel_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-missing")
    assert (
        verify_variant(
            variant_path,
            policies=TEST_POLICIES,
        )
        == VerificationResult.SignatureBundleMissing()
    )


def test_broken_signature_bundle_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-broken")
    match verify_variant(
        variant_path,
        policies=TEST_POLICIES,
    ):
        case VerificationResult.SignatureBundleInvalid(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureBundleInvalid, was: {other}")


def test_invalid_signature_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-invalid")
    match verify_variant(
        variant_path,
        policies=TEST_POLICIES,
    ):
        case VerificationResult.SignatureVerificationFailure(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")
