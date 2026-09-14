from dataclasses import is_dataclass
from pathlib import Path

import pytest
from sigstore.verify import policy

import kernels.verify as verify_module
from kernels import install_kernel
from kernels._rust import DigestViolation, KernelLocation, Oid
from kernels._versions import resolve_revision_or_version
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
def signed_kernel():
    """A correctly signed kernel, with the location that identifies it."""
    repo_id = "kernels-test/signatures"
    revision = resolve_revision_or_version(repo_id, revision=None, version=1, local_files_only=False)
    variant_path = install_kernel(repo_id, revision=str(revision))
    return variant_path, KernelLocation.remote(repo_id, revision, variant_path.name)


def _verify_uncached(variant_path: Path, **kwargs) -> VerificationResult.Any:
    """Verify a variant without reading or writing the receipt cache.

    Used by the tests that exercise verification itself rather than caching,
    both to keep them away from the real receipt store and because a location
    is required but unused when caching is off.
    """
    return verify_variant(
        variant_path,
        location=KernelLocation.remote("kernels-test/signatures", Oid.from_str("0" * 40), variant_path.name),
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
    revision = resolve_revision_or_version("kernels-community/relu", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-community/relu", revision=str(revision))
    assert _verify_uncached(variant_path) == VerificationResult.Success()


def test_correctly_signed_kernel_passes():
    revision = resolve_revision_or_version("kernels-test/signatures", revision=None, version=1, local_files_only=False)
    variant_path = install_kernel("kernels-test/signatures", revision=str(revision))
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
                    cache_dir=CACHE_DIR,
                    revision=str(revision),
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
                    cache_dir=CACHE_DIR,
                    revision=str(revision),
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


ALL_RESULTS = [
    VerificationResult.Success(),
    VerificationResult.SignatureBundleMissing(),
    VerificationResult.SignatureBundleInvalid(reason="bad bundle"),
    VerificationResult.SignatureVerificationFailure(reason="bad signature"),
    VerificationResult.MetadataMissing(),
    VerificationResult.MetadataInvalid(reason="bad metadata"),
    VerificationResult.DigestMissing(),
    VerificationResult.DigestVerificationFailure(violations=[DigestViolation.MissingFile("kernel.py")]),
]


def test_all_results_are_covered():
    """`ALL_RESULTS` must cover every variant.

    Without this, adding a variant would silently skip the tests below, which
    is exactly when they are needed.
    """
    variants = {
        name for name, member in vars(VerificationResult).items() if isinstance(member, type) and is_dataclass(member)
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
    is_success = isinstance(result, VerificationResult.Success)
    assert isinstance(result, VerificationResult.Failure) != is_success


def test_result_messages_include_their_detail():
    assert "bang" in str(VerificationResult.SignatureBundleInvalid(reason="bang"))
    assert "bang" in str(VerificationResult.MetadataInvalid(reason="bang"))
    assert "bang" in str(VerificationResult.SignatureVerificationFailure(reason="bang"))

    violations = [DigestViolation.MissingFile("kernel.py"), DigestViolation.UnknownFile("extra.so")]
    message = str(VerificationResult.DigestVerificationFailure(violations=violations))
    for violation in violations:
        assert str(violation) in message


def test_failure_must_describe_itself():
    """The base class makes a message mandatory for new failures."""

    class Undescribed(VerificationResult.Failure):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Undescribed()  # type: ignore[abstract]
