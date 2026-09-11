import logging
from pathlib import Path

import pytest
from sigstore.verify import policy

import kernels.verify as verify_module
from kernels import install_kernel
from kernels._rust import DigestViolation, KernelLocation, ReceiptStore
from kernels._versions import select_revision_or_version
from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS
from kernels.verify import VerificationResult, verify_variant

TEST_POLICY: policy.VerificationPolicy = policy.Identity(
    identity="me@danieldk.eu", issuer="https://github.com/login/oauth"
)

OTHER_POLICY: policy.VerificationPolicy = policy.Identity(
    identity="nobody@example.com", issuer="https://github.com/login/oauth"
)


@pytest.fixture
def receipt_store(tmp_path, monkeypatch):
    """An isolated receipt store, so that tests do not share verifications."""
    receipt_dir = tmp_path / "receipts"
    store = ReceiptStore.from_path(receipt_dir)
    monkeypatch.setattr(verify_module, "_open_receipt_store", lambda: store)
    return store


@pytest.fixture
def signed_kernel():
    """A correctly signed kernel, with the location that identifies it."""
    repo_id = "kernels-test/signatures"
    revision = select_revision_or_version(repo_id, revision=None, version=1, local_files_only=False)
    variant_path = install_kernel(repo_id, revision=revision)
    return variant_path, KernelLocation.remote(repo_id, revision, variant_path.name)


def _verify_uncached(variant_path: Path, **kwargs) -> VerificationResult.Any:
    """Verify a variant without reading or writing the receipt cache.

    Used by the tests that exercise verification itself rather than caching,
    both to keep them away from the real receipt store and because a location
    is required but unused when caching is off.
    """
    return verify_variant(
        variant_path,
        location=KernelLocation.remote("kernels-test/signatures", "0" * 40, variant_path.name),
        cache=False,
        **kwargs,
    )


def _no_hashing(monkeypatch):
    """Make rehashing the variant fail, so that only cache hits can succeed."""

    class ExplodingDigest:
        @staticmethod
        def hash_variant(*args, **kwargs):
            raise AssertionError("the variant was rehashed, so this was not a cache hit")

    # Patch the name in `kernels.verify`: `Digest` is an extension type, whose
    # attributes cannot be set.
    monkeypatch.setattr(verify_module, "Digest", ExplodingDigest)


def test_correctly_signed_kernel_passes_with_default_policy():
    revision = select_revision_or_version("kernels-community/relu", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-community/relu", revision=revision)
    assert _verify_uncached(variant_path) == VerificationResult.Success()


def test_correctly_signed_kernel_passes():
    revision = select_revision_or_version("kernels-test/signatures", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-test/signatures", revision=revision)
    assert _verify_uncached(variant_path, policy=TEST_POLICY) == VerificationResult.Success()


def test_invalid_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="invalid-digest")

    match _verify_uncached(variant_path, policy=TEST_POLICY):
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

    match _verify_uncached(
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
    assert _verify_uncached(variant_path, policy=TEST_POLICY) == VerificationResult.DigestMissing()


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
        _verify_uncached(
            # No CUDA dependency, we are only checking metadata.
            variant_paths / "torch-cuda",
            policy=TEST_POLICY,
        )
        == VerificationResult.MetadataMissing()
    )


def test_unsigned_kernel_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-missing")
    assert _verify_uncached(variant_path, policy=TEST_POLICY) == VerificationResult.SignatureBundleMissing()


def test_broken_signature_bundle_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-broken")
    match _verify_uncached(variant_path, policy=TEST_POLICY):
        case VerificationResult.SignatureBundleInvalid(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureBundleInvalid, was: {other}")


def test_invalid_signature_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-invalid")
    match _verify_uncached(variant_path, policy=TEST_POLICY):
        case VerificationResult.SignatureVerificationFailure(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")


def test_verification_is_cached(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()
    assert receipt_store.load(location) is not None

    # The second verification must be served from the receipt, without
    # rehashing the variant.
    _no_hashing(monkeypatch)
    assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()


def test_verification_is_not_cached_with_cache_off(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    result = verify_variant(variant_path, policy=TEST_POLICY, location=location, cache=False)
    assert result == VerificationResult.Success()

    # Nothing was recorded, ...
    assert receipt_store.load(location) is None

    # ... and a verification with caching off does the full work even when a
    # receipt does exist.
    assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()
    assert receipt_store.load(location) is not None

    _no_hashing(monkeypatch)
    with pytest.raises(AssertionError, match="was rehashed"):
        verify_variant(variant_path, policy=TEST_POLICY, location=location, cache=False)


def test_cached_verification_still_enforces_policy(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    # Verify under a policy that accepts this kernel, so a receipt is stored.
    assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()

    # The receipt says the kernel was verified, but not *under which policy*,
    # so a policy that does not accept this signer must still reject it.
    _no_hashing(monkeypatch)
    match verify_variant(variant_path, policy=OTHER_POLICY, location=location):
        case VerificationResult.SignatureVerificationFailure():
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")


def test_unusable_receipt_falls_back_to_verification(receipt_store, signed_kernel, tmp_path, caplog):
    variant_path, location = signed_kernel

    assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()

    (receipt_path,) = list((tmp_path / "receipts").iterdir())
    receipt_path.write_text("not a receipt")

    with caplog.at_level(logging.WARNING, logger="kernels.verify"):
        assert verify_variant(variant_path, policy=TEST_POLICY, location=location) == VerificationResult.Success()

    assert "unusable kernel verification receipt" in caplog.text
