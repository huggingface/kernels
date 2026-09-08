import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from cryptography import x509
from cryptography.hazmat.primitives.serialization import Encoding
from huggingface_hub import constants

from kernels.hf_hub import CACHE_DIR

if TYPE_CHECKING:
    from kernels.resolver import LocalKernel

CACHE_FORMAT_VERSION = "v1"


class _Hash(Protocol):
    def update(self, data: bytes) -> None: ...


def _update(hash: _Hash, part: str) -> None:
    """Update the hash with a string, unambiguously (length-prefixed)."""
    encoded = part.encode()
    hash.update(len(encoded).to_bytes(8, "big"))
    hash.update(encoded)


def _fingerprint_files(variant_path: Path, hash: _Hash) -> None:
    """Update the hash with the metadata of every file in the variant.

    File content is never read: the fingerprint consists of paths, sizes, and
    modification times, so it is cheap even for large kernels.
    """
    entries = []
    for root, _dirs, files in os.walk(variant_path):
        for name in files:
            path = Path(root) / name
            stat = path.stat()
            entries.append((str(path.relative_to(variant_path)), stat.st_size, stat.st_mtime_ns))
    entries.sort()

    for relpath, size, mtime_ns in entries:
        _update(hash, relpath)
        hash.update(size.to_bytes(8, "big"))
        hash.update(mtime_ns.to_bytes(8, "big", signed=True))


def receipt_key(kernel: "LocalKernel") -> str:
    """The receipt filename for the given kernel.

    Hub kernels are identified by repository, commit, and variant. Local
    kernels by their path, file metadata fingerprint, and variant.
    """
    hash = hashlib.sha256()
    if kernel.origin is not None:
        _update(hash, "hub")
        _update(hash, kernel.origin.repo_id)
        _update(hash, kernel.origin.revision)
    else:
        _update(hash, "local")
        _update(hash, str(kernel.variant_path.resolve()))
        _fingerprint_files(kernel.variant_path, hash)
    _update(hash, kernel.variant_str)
    return hash.hexdigest()


@dataclass
class VerificationReceipt:
    """Record of a successful kernel verification."""

    certificate: x509.Certificate
    """The signing certificate from the sigstore bundle."""

    variant: str
    """The verified variant."""

    digest: str
    """The verified kernel digest (informational, not used on cache hits)."""

    verified_at: datetime
    """When the kernel was verified."""

    repo_id: str | None = None
    """Repository of the verified kernel (Hub kernels only, informational)."""

    revision: str | None = None
    """Commit of the verified kernel (Hub kernels only, informational)."""

    path: str | None = None
    """Path of the verified kernel (local kernels only, informational)."""


def _receipt_dir() -> Path:
    # Same (import-time) cache resolution as kernel downloads.
    return Path(CACHE_DIR or constants.HF_HUB_CACHE) / ".verified-kernel" / CACHE_FORMAT_VERSION


def load_receipt(key: str) -> VerificationReceipt | None:
    """Load a receipt, or `None` if it is missing or corrupt."""
    try:
        data = json.loads((_receipt_dir() / key).read_bytes())
        return VerificationReceipt(
            certificate=x509.load_der_x509_certificate(base64.b64decode(data["certificate"])),
            variant=data["variant"],
            digest=data["digest"],
            verified_at=datetime.fromisoformat(data["verified_at"]),
            repo_id=data.get("repo_id"),
            revision=data.get("revision"),
            path=data.get("path"),
        )
    except (OSError, ValueError, KeyError):
        return None


def store_receipt(key: str, receipt: VerificationReceipt) -> None:
    """Store a receipt atomically. Raises `OSError` if the cache is not writable."""
    receipt_dir = _receipt_dir()
    receipt_dir.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        {
            "certificate": base64.b64encode(receipt.certificate.public_bytes(Encoding.DER)).decode(),
            "variant": receipt.variant,
            "digest": receipt.digest,
            "verified_at": receipt.verified_at.isoformat(),
            "repo_id": receipt.repo_id,
            "revision": receipt.revision,
            "path": receipt.path,
        }
    )

    fd, tmp_path = tempfile.mkstemp(dir=receipt_dir, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as tmp_file:
            tmp_file.write(payload)
        os.replace(tmp_path, receipt_dir / key)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
