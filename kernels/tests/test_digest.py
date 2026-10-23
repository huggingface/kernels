import json
import logging
import subprocess
import sys
from dataclasses import is_dataclass

import pytest

import kernels.digest as digest_module
from kernels._rust import (
    Digest,
    DigestAlgorithm,
    DigestReceiptStore,
    DigestViolation,
    KernelLocation,
    Metadata,
    Oid,
)
from kernels.digest import DigestVerificationResult, verify_digest

_VARIANT = "torch-cpu"


def _write_variant(tmp_path, *, with_digest: bool = True):
    """Write a kernel variant, with a digest of its files in the metadata."""
    variant_path = tmp_path / _VARIANT
    variant_path.mkdir()
    (variant_path / "__init__.py").write_text("from ._ops import ops\n")
    (variant_path / "_ops.py").write_text("ops = None\n")
    (variant_path / "_kernel.abi3.so").write_bytes(b"\x7fELF kernel")

    metadata = {
        "name": "test-kernel",
        "id": "_test_kernel_cpu_abc123",
        "version": 1,
        "license": "mit",
        "python-depends": [],
        "backend": {"type": "cpu"},
    }
    if with_digest:
        metadata["digest"] = {
            "algorithm": "sha256",
            "files": Digest.hash_variant(DigestAlgorithm.SHA256, variant_path).files,
        }
    (variant_path / "metadata.json").write_text(json.dumps(metadata))

    return variant_path


def _metadata(variant_path) -> Metadata:
    return Metadata.read_from_file(variant_path / "metadata.json")


def _location(variant_path) -> KernelLocation:
    return KernelLocation.remote("kernels-test/digest", Oid.from_str("a" * 40), variant_path.name)


@pytest.fixture
def receipt_store(tmp_path, monkeypatch):
    """An isolated receipt store, so that tests do not share verifications."""
    store = DigestReceiptStore.from_path(tmp_path / "receipts")
    monkeypatch.setattr(digest_module, "_open_digest_receipt_store", lambda: store)
    return store


def _no_hashing(monkeypatch):
    """Make rehashing the variant fail, so that only cache hits can succeed."""

    class ExplodingDigest:
        @staticmethod
        def hash_variant(*args, **kwargs):
            raise AssertionError("the variant was rehashed, so this was not a cache hit")

    # Patch the name in `kernels.digest`: `Digest` is an extension type, whose
    # attributes cannot be set.
    monkeypatch.setattr(digest_module, "Digest", ExplodingDigest)


def test_matching_files_pass(tmp_path):
    variant_path = _write_variant(tmp_path)
    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=None)
    assert result == DigestVerificationResult.Success()


def test_modified_file_fails(tmp_path):
    variant_path = _write_variant(tmp_path)
    (variant_path / "_ops.py").write_text("ops = 'hacked'\n")

    match verify_digest(variant_path, metadata=_metadata(variant_path), location=None):
        case DigestVerificationResult.DigestVerificationFailure(
            violations=[DigestViolation.HashMismatch() as violation]
        ):
            assert violation.path == "_ops.py"
        case other:
            raise RuntimeError(f"Expected a single hash mismatch, was: {other}")


def test_added_file_fails(tmp_path):
    variant_path = _write_variant(tmp_path)
    (variant_path / "extra.py").write_text("print('hi')\n")

    match verify_digest(variant_path, metadata=_metadata(variant_path), location=None):
        case DigestVerificationResult.DigestVerificationFailure(
            violations=[DigestViolation.UnknownFile() as violation]
        ):
            assert violation.path == "extra.py"
        case other:
            raise RuntimeError(f"Expected a single unknown file, was: {other}")


def test_removed_file_fails(tmp_path):
    variant_path = _write_variant(tmp_path)
    (variant_path / "_ops.py").unlink()

    match verify_digest(variant_path, metadata=_metadata(variant_path), location=None):
        case DigestVerificationResult.DigestVerificationFailure(
            violations=[DigestViolation.MissingFile() as violation]
        ):
            assert violation.path == "_ops.py"
        case other:
            raise RuntimeError(f"Expected a single missing file, was: {other}")


def test_bytecode_is_ignored(tmp_path):
    variant_path = _write_variant(tmp_path)
    pycache = variant_path / "__pycache__"
    pycache.mkdir()
    (pycache / "_ops.cpython-314.pyc").write_bytes(b"bytecode")

    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=None)
    assert result == DigestVerificationResult.Success()


def test_missing_digest(tmp_path):
    variant_path = _write_variant(tmp_path, with_digest=False)
    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=None)
    assert result == DigestVerificationResult.DigestMissing()


def test_verification_is_cached(tmp_path, receipt_store, monkeypatch):
    variant_path = _write_variant(tmp_path)
    location = _location(variant_path)

    assert verify_digest(variant_path, metadata=_metadata(variant_path), location=location) == (
        DigestVerificationResult.Success()
    )
    assert receipt_store.load(location) is not None

    # The second verification must be served from the receipt, without
    # rehashing the variant.
    _no_hashing(monkeypatch)
    assert verify_digest(variant_path, metadata=_metadata(variant_path), location=location) == (
        DigestVerificationResult.Success()
    )


def test_failed_verification_is_not_cached(tmp_path, receipt_store):
    variant_path = _write_variant(tmp_path)
    (variant_path / "_ops.py").write_text("ops = 'hacked'\n")
    location = _location(variant_path)

    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=location)
    assert isinstance(result, DigestVerificationResult.DigestVerificationFailure)
    assert receipt_store.load(location) is None


def test_verification_is_not_cached_with_cache_off(tmp_path, receipt_store, monkeypatch):
    variant_path = _write_variant(tmp_path)
    location = _location(variant_path)

    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=location, cache=False)
    assert result == DigestVerificationResult.Success()

    # Nothing was recorded, ...
    assert receipt_store.load(location) is None

    # ... and a verification with caching off does the full work even when a
    # receipt does exist.
    assert verify_digest(variant_path, metadata=_metadata(variant_path), location=location) == (
        DigestVerificationResult.Success()
    )
    assert receipt_store.load(location) is not None

    _no_hashing(monkeypatch)
    with pytest.raises(AssertionError, match="was rehashed"):
        verify_digest(variant_path, metadata=_metadata(variant_path), location=location, cache=False)


def test_kernel_without_location_does_not_use_receipts(tmp_path, monkeypatch):
    variant_path = _write_variant(tmp_path)

    def no_store():
        raise AssertionError("the receipt store was opened for a kernel without a location")

    monkeypatch.setattr(digest_module, "_open_digest_receipt_store", no_store)

    result = verify_digest(variant_path, metadata=_metadata(variant_path), location=None)
    assert result == DigestVerificationResult.Success()


def test_unusable_receipt_falls_back_to_verification(tmp_path, receipt_store, caplog):
    variant_path = _write_variant(tmp_path)
    location = _location(variant_path)

    assert verify_digest(variant_path, metadata=_metadata(variant_path), location=location) == (
        DigestVerificationResult.Success()
    )

    (receipt_path,) = list((tmp_path / "receipts").iterdir())
    receipt_path.write_text("not a receipt")

    with caplog.at_level(logging.WARNING, logger="kernels.digest"):
        assert verify_digest(variant_path, metadata=_metadata(variant_path), location=location) == (
            DigestVerificationResult.Success()
        )

    assert "unusable kernel verification receipt" in caplog.text


def test_digest_verification_does_not_need_sigstore():
    """Digest verification must work when the optional `sigstore` is not installed."""
    script = (
        "import sys\n"
        # A `None` entry makes importing the module fail.
        "sys.modules['sigstore'] = None\n"
        "import kernels.digest, kernels.validate\n"
        "from kernels.compat import has_sigstore\n"
        "assert not has_sigstore\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


ALL_RESULTS = [
    DigestVerificationResult.Success(),
    DigestVerificationResult.DigestMissing(),
    DigestVerificationResult.DigestVerificationFailure(violations=[DigestViolation.MissingFile("kernel.py")]),
]


def test_all_results_are_covered():
    """`ALL_RESULTS` must cover every variant.

    Without this, adding a variant would silently skip the tests below, which
    is exactly when they are needed.
    """
    variants = {
        name
        for name, member in vars(DigestVerificationResult).items()
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
    is_success = isinstance(result, DigestVerificationResult.Success)
    assert isinstance(result, DigestVerificationResult.Failure) != is_success


def test_result_messages_include_their_detail():
    violations = [DigestViolation.MissingFile("kernel.py"), DigestViolation.UnknownFile("extra.so")]
    message = str(DigestVerificationResult.DigestVerificationFailure(violations=violations))
    for violation in violations:
        assert str(violation) in message


def test_failure_must_describe_itself():
    """The base class makes a message mandatory for new failures."""

    class Undescribed(DigestVerificationResult.Failure):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Undescribed()  # type: ignore[abstract]
