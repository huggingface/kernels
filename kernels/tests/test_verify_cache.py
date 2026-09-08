from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from kernels_data import Metadata

import kernels.verify_cache as verify_cache
from kernels.resolver import LocalKernel, RemoteKernel
from kernels.variants import parse_variant
from kernels.verify_cache import VerificationReceipt, load_receipt, receipt_key, store_receipt


@pytest.fixture
def receipt_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_cache, "_receipt_dir", lambda: tmp_path)
    return tmp_path


def _local_kernel(variant_path: Path, metadata: Metadata) -> LocalKernel:
    return LocalKernel(variant_path=variant_path, metadata=metadata)


def _hub_kernel(variant_path: Path, metadata: Metadata) -> LocalKernel:
    return LocalKernel(
        variant_path=variant_path,
        metadata=metadata,
        origin=RemoteKernel(
            repo_id="kernels-test/signatures",
            revision="a" * 40,
            metadata=metadata,
            variant=parse_variant("torch-cuda"),
        ),
    )


def _receipt(certificate: x509.Certificate, **overrides) -> VerificationReceipt:
    fields = {
        "certificate": certificate,
        "variant": "test-variant",
        "digest": "test-digest",
        "verified_at": datetime.now(timezone.utc),
    }
    fields.update(overrides)
    return VerificationReceipt(**fields)


def test_receipt_key_is_deterministic_and_distinguishes_origin(tmp_path, make_metadata):
    metadata = make_metadata("cuda", None)
    local = _local_kernel(tmp_path / "test-variant", metadata)
    hub = _hub_kernel(tmp_path / "test-variant", metadata)

    assert receipt_key(local) == receipt_key(local)
    assert receipt_key(hub) == receipt_key(hub)
    assert receipt_key(local) != receipt_key(hub)


def test_local_receipt_key_changes_when_files_change(tmp_path, make_metadata):
    variant_path = tmp_path / "test-variant"
    variant_path.mkdir()
    (variant_path / "kernel.py").write_bytes(b"pass")

    key_before = receipt_key(_local_kernel(variant_path, make_metadata("cuda", None)))

    # Different size, so the fingerprint changes even at equal mtime granularity.
    (variant_path / "kernel.py").write_bytes(b"pass  # rebuilt")

    assert receipt_key(_local_kernel(variant_path, make_metadata("cuda", None))) != key_before


def test_receipt_roundtrip(receipt_dir, test_certificate):
    key = "a" * 64
    store_receipt(key, _receipt(test_certificate, repo_id="kernels-test/signatures", revision="b" * 40))

    receipt = load_receipt(key)
    assert receipt is not None
    assert receipt.certificate.public_bytes(Encoding.DER) == test_certificate.public_bytes(Encoding.DER)
    assert receipt.variant == "test-variant"
    assert receipt.digest == "test-digest"
    assert receipt.repo_id == "kernels-test/signatures"
    assert receipt.revision == "b" * 40
    assert receipt.path is None


def test_load_receipt_missing_or_corrupt(receipt_dir):
    assert load_receipt("a" * 64) is None

    (receipt_dir / ("a" * 64)).write_text("not a receipt")
    assert load_receipt("a" * 64) is None


def test_store_receipt_raises_when_cache_not_writable(receipt_dir, test_certificate):
    # A regular file where the receipt directory should be.
    receipt_dir.rmdir()
    receipt_dir.write_text("not a directory")

    with pytest.raises(OSError):
        store_receipt("a" * 64, _receipt(test_certificate))
