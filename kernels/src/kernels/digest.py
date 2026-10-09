import abc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, final

from kernels._rust import (
    Digest,
    DigestReceipt,
    DigestReceiptStore,
    DigestValidationError,
    DigestViolation,
    KernelLocation,
    Metadata,
    ReceiptError,
    SignatureReceiptStore,
)

logger = logging.getLogger(__name__)


class DigestVerificationResult:
    class Failure(abc.ABC):
        """A kernel build variant whose files could not be verified against its digest."""

        @abc.abstractmethod
        def __str__(self) -> str: ...

    @final
    @dataclass
    class DigestVerificationFailure(Failure):
        """
        Verification failed because there were digest violations.

        The violations are provided through the `violations` field.
        """

        violations: list[DigestViolation]

        def __str__(self) -> str:
            violations = "\n".join(str(violation) for violation in self.violations)
            return f"the files do not match the digest in the metadata, so they may have been modified:\n{violations}"

    @final
    @dataclass
    class DigestMissing(Failure):
        """
        Verification failed because the metadata did not have a digest.
        """

        def __str__(self) -> str:
            return "the metadata does not record a digest, so its integrity cannot be verified"

    @final
    @dataclass
    class Success:
        """
        Verification was successful.
        """

        def __str__(self) -> str:
            return "the files match the digest in the metadata"

    Any: TypeAlias = DigestMissing | DigestVerificationFailure | Success


def _open_digest_receipt_store() -> DigestReceiptStore | None:
    """The digest receipt store, or `None` when verifications cannot be cached."""
    try:
        return DigestReceiptStore.in_kernels_cache()
    except ReceiptError as e:
        logger.warning(f"Cannot cache kernel digest verifications: {e}")
        return None


def _has_receipt(store: DigestReceiptStore | SignatureReceiptStore, location: KernelLocation) -> bool:
    """Whether the kernel at `location` was verified before.

    An unusable receipt counts as a cache miss: the kernel is then verified in
    full, which overwrites the receipt. A broken cache must never make a kernel
    fail to verify.
    """
    try:
        return store.load(location) is not None
    except ReceiptError as e:
        logger.warning(f"Ignoring unusable kernel verification receipt: {e}")
        return False


def verify_digest(
    variant_path: Path,
    *,
    metadata: Metadata,
    location: KernelLocation | None,
    cache: bool = True,
) -> DigestVerificationResult.Any:
    """
    Verify the files of a kernel variant against the digest in its metadata.

    This only checks that the files and the metadata agree. It does not check
    the authenticity of the metadata, use `kernels.verify.verify_signature`
    for that.

    Args:
        variant_path (`Path`):
            Kernel variant path.
        metadata (`Metadata`):
            The metadata of the kernel variant.
        location (`KernelLocation`, *optional*):
            Identity of the kernel, used to cache the verification. Kernels
            without a location (e.g. local kernels) are always verified in full.
        cache (`bool`):
            Whether to use the receipt cache to lookup or store kernel
            verifications.
    """
    if metadata.digest is None:
        return DigestVerificationResult.DigestMissing()

    receipt_store = None
    if cache and location is not None:
        receipt_store = _open_digest_receipt_store()
        if receipt_store is not None and _has_receipt(receipt_store, location):
            return DigestVerificationResult.Success()

    # The validation is delegated to the (Rust) `Digest.validate`, which raises
    # with each individual violation.
    current_digest = Digest.hash_variant(metadata.digest.algorithm, variant_path)
    try:
        metadata.digest.validate(current_digest)
    except DigestValidationError as e:
        return DigestVerificationResult.DigestVerificationFailure(violations=e.violations)

    if receipt_store is not None and location is not None:
        try:
            receipt_store.store(DigestReceipt(location))
        except ReceiptError as e:
            logger.warning(f"Cannot store kernel digest verification receipt: {e}")

    return DigestVerificationResult.Success()
