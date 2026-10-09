import logging
from dataclasses import is_dataclass
from pathlib import Path

import pytest
from sigstore.verify import policy

import kernels.verify as verify_module
from kernels import install_kernel
from kernels._rust import DigestViolation, KernelLocation, Metadata, Oid, SignatureReceiptStore
from kernels._versions import resolve_revision_or_version
from kernels.digest import DigestVerificationResult, verify_digest
from kernels.hf_hub import _get_cache_dir, _get_hf_api
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS
from kernels.verify import SignatureVerificationResult, verify_signature

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
    store = SignatureReceiptStore.from_path(receipt_dir)
    monkeypatch.setattr(verify_module, "_open_signature_receipt_store", lambda: store)
    return store


@pytest.fixture
def signed_kernel():
    """A correctly signed kernel, with the location that identifies it."""
    repo_id = "kernels-test/signatures"
    revision = resolve_revision_or_version(repo_id, revision=None, version=1, local_files_only=False)
    variant_path = install_kernel(repo_id, revision=str(revision))
    return variant_path, KernelLocation.remote(repo_id, revision, variant_path.name)


def _verify_signature_uncached(variant_path: Path, **kwargs) -> SignatureVerificationResult.Any:
    """Verify the signature of a variant without reading or writing the receipt cache.

    Used by the tests that exercise verification itself rather than caching,
    both to keep them away from the real receipt store and because a location
    is required but unused when caching is off.
    """
    return verify_signature(
        variant_path,
        location=KernelLocation.remote("kernels-test/signatures", Oid.from_str("0" * 40), variant_path.name),
        cache=False,
        **kwargs,
    )


def _verify_digest_uncached(variant_path: Path) -> DigestVerificationResult.Any:
    """Verify the files of a variant against its digest, without receipts."""
    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    return verify_digest(variant_path, metadata=metadata, location=None, cache=False)


def _no_signature_verification(monkeypatch):
    """Make full signature verification fail, so that only cache hits can succeed."""

    class ExplodingVerifier:
        @staticmethod
        def production(*args, **kwargs):
            raise AssertionError("the signature was verified in full, so this was not a cache hit")

    monkeypatch.setattr(verify_module, "Verifier", ExplodingVerifier)


def test_correctly_signed_kernel_passes_with_default_policy():
    revision = resolve_revision_or_version("kernels-community/relu", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-community/relu", revision=str(revision))
    assert _verify_signature_uncached(variant_path) == SignatureVerificationResult.Success()
    assert _verify_digest_uncached(variant_path) == DigestVerificationResult.Success()


def test_correctly_signed_kernel_passes():
    revision = resolve_revision_or_version("kernels-test/signatures", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-test/signatures", revision=str(revision))
    assert _verify_signature_uncached(variant_path, policy=TEST_POLICY) == SignatureVerificationResult.Success()
    assert _verify_digest_uncached(variant_path) == DigestVerificationResult.Success()


def test_invalid_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="invalid-digest")

    # The metadata itself is correctly signed, signature verification does
    # not check the files.
    assert _verify_signature_uncached(variant_path, policy=TEST_POLICY) == SignatureVerificationResult.Success()

    match _verify_digest_uncached(variant_path):
        case DigestVerificationResult.DigestVerificationFailure(violations=violations):
            assert len(violations) == 1
            assert isinstance(violations[0], DigestViolation.HashMismatch)
        case other:
            raise RuntimeError(f"Expected DigestVerificationFailure, was: {other}")


def test_invalid_metadata_fails():
    # We cannot use regular code paths, because they require valid metadata.
    revision = resolve_revision_or_version(
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
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                )
            )
        )
        / "build"
    )

    match _verify_signature_uncached(
        # No CUDA dependency, we are only checking metadata.
        variant_paths / "torch-cuda",
        policy=TEST_POLICY,
    ):
        case SignatureVerificationResult.MetadataInvalid(reason=reason):
            assert "Cannot parse metadata" in reason
        case other:
            raise RuntimeError(f"Expected MetadataInvalid, was: {other}")


def test_missing_digest_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="missing-digest")
    assert _verify_signature_uncached(variant_path, policy=TEST_POLICY) == SignatureVerificationResult.Success()
    assert _verify_digest_uncached(variant_path) == DigestVerificationResult.DigestMissing()


def test_missing_metadata_fails():
    # We cannot use regular code paths, because they require valid metadata.
    revision = resolve_revision_or_version(
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
                    cache_dir=_get_cache_dir(),
                    revision=str(revision),
                )
            )
        )
        / "build"
    )

    assert (
        _verify_signature_uncached(
            # No CUDA dependency, we are only checking metadata.
            variant_paths / "torch-cuda",
            policy=TEST_POLICY,
        )
        == SignatureVerificationResult.MetadataMissing()
    )


def test_unsigned_kernel_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-missing")
    assert (
        _verify_signature_uncached(variant_path, policy=TEST_POLICY)
        == SignatureVerificationResult.SignatureBundleMissing()
    )


def test_broken_signature_bundle_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-broken")
    match _verify_signature_uncached(variant_path, policy=TEST_POLICY):
        case SignatureVerificationResult.SignatureBundleInvalid(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureBundleInvalid, was: {other}")


def test_invalid_signature_fails():
    variant_path = install_kernel("kernels-test/signatures", revision="signature-invalid")
    match _verify_signature_uncached(variant_path, policy=TEST_POLICY):
        case SignatureVerificationResult.SignatureVerificationFailure(reason=_):
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")


def test_verification_is_cached(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    assert (
        verify_signature(variant_path, policy=TEST_POLICY, location=location) == SignatureVerificationResult.Success()
    )
    assert receipt_store.load(location) is not None

    # The second verification must be served from the receipt, without
    # verifying the signature in full.
    _no_signature_verification(monkeypatch)
    assert (
        verify_signature(variant_path, policy=TEST_POLICY, location=location) == SignatureVerificationResult.Success()
    )


def test_verification_is_not_cached_with_cache_off(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    result = verify_signature(variant_path, policy=TEST_POLICY, location=location, cache=False)
    assert result == SignatureVerificationResult.Success()

    # Nothing was recorded, ...
    assert receipt_store.load(location) is None

    # ... and a verification with caching off does the full work even when a
    # receipt does exist.
    assert (
        verify_signature(variant_path, policy=TEST_POLICY, location=location) == SignatureVerificationResult.Success()
    )
    assert receipt_store.load(location) is not None

    _no_signature_verification(monkeypatch)
    with pytest.raises(AssertionError, match="verified in full"):
        verify_signature(variant_path, policy=TEST_POLICY, location=location, cache=False)


def test_cached_verification_still_enforces_policy(receipt_store, signed_kernel, monkeypatch):
    variant_path, location = signed_kernel

    # Verify under a policy that accepts this kernel, so a receipt is stored.
    assert (
        verify_signature(variant_path, policy=TEST_POLICY, location=location) == SignatureVerificationResult.Success()
    )

    # The receipt says the kernel was verified, but not *under which policy*,
    # so a policy that does not accept this signer must still reject it.
    _no_signature_verification(monkeypatch)
    match verify_signature(variant_path, policy=OTHER_POLICY, location=location):
        case SignatureVerificationResult.SignatureVerificationFailure():
            pass
        case other:
            raise RuntimeError(f"Expected SignatureVerificationFailure, was: {other}")


def test_unusable_receipt_falls_back_to_verification(receipt_store, signed_kernel, tmp_path, caplog):
    variant_path, location = signed_kernel

    assert (
        verify_signature(variant_path, policy=TEST_POLICY, location=location) == SignatureVerificationResult.Success()
    )

    (receipt_path,) = list((tmp_path / "receipts").iterdir())
    receipt_path.write_text("not a receipt")

    with caplog.at_level(logging.WARNING):
        assert (
            verify_signature(variant_path, policy=TEST_POLICY, location=location)
            == SignatureVerificationResult.Success()
        )

    assert "unusable kernel verification receipt" in caplog.text


ALL_RESULTS = [
    SignatureVerificationResult.Success(),
    SignatureVerificationResult.SignatureBundleMissing(),
    SignatureVerificationResult.SignatureBundleInvalid(reason="bad bundle"),
    SignatureVerificationResult.SignatureVerificationFailure(reason="bad signature"),
    SignatureVerificationResult.MetadataMissing(),
    SignatureVerificationResult.MetadataInvalid(reason="bad metadata"),
]


def test_all_results_are_covered():
    """`ALL_RESULTS` must cover every variant.

    Without this, adding a variant would silently skip the tests below, which
    is exactly when they are needed.
    """
    variants = {
        name
        for name, member in vars(SignatureVerificationResult).items()
        if isinstance(member, type) and is_dataclass(member)
    }
    assert {type(result).__name__ for result in ALL_RESULTS} == variants


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda result: type(result).__name__)
def test_every_result_describes_itself(result):
    message = str(result)
    assert message
    # Prose, rather than the dataclass repr that `str` falls back to.
    assert message != repr(result)
    assert not message.startswith(type(result).__name__)


@pytest.mark.parametrize("result", ALL_RESULTS, ids=lambda result: type(result).__name__)
def test_only_success_is_not_a_failure(result):
    is_success = isinstance(result, SignatureVerificationResult.Success)
    assert isinstance(result, SignatureVerificationResult.Failure) != is_success


def test_result_messages_include_their_detail():
    assert "bang" in str(SignatureVerificationResult.SignatureBundleInvalid(reason="bang"))
    assert "bang" in str(SignatureVerificationResult.MetadataInvalid(reason="bang"))
    assert "bang" in str(SignatureVerificationResult.SignatureVerificationFailure(reason="bang"))


def test_failure_must_describe_itself():
    """The base class makes a message mandatory for new failures."""

    class Undescribed(SignatureVerificationResult.Failure):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Undescribed()  # type: ignore[abstract]
