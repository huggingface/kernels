import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from packaging.version import InvalidVersion, parse

from kernels._data import Metadata, Version
from kernels.archs import _check_arch_incompatibility
from kernels.backends import _backend
from kernels.python_deps import validate_dependencies

from .compat import has_sigstore

if TYPE_CHECKING:
    from sigstore.verify.policy import VerificationPolicy

    from kernels.resolver import LocalKernel
    from kernels.verify_cache import VerificationReceipt

logger = logging.getLogger(__name__)


# Metadata validators.


class MetadataValidator(Protocol):
    """Kernel metadata validator."""

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None: ...


class DependencyValidator:
    """Validate the Python dependencies of a kernel build variant."""

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        validate_dependencies(
            metadata.name.python_name,
            metadata.python_depends,
            _backend(),
        )


class ArchValidator:
    """Validate architecture compatibility for a kernel build variant."""

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        _check_arch_incompatibility(metadata, variant)


class DirtyValidator:
    """Warn when a kernel variant was built from a dirty git tree."""

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        provenance = metadata.provenance
        if provenance is None or not provenance.dirty:
            return

        dirty_sources = []
        if provenance.kernel is not None and provenance.kernel.dirty:
            dirty_sources.append("kernel source")
        builder_git = provenance.kernel_builder.git
        if builder_git is not None and builder_git.dirty:
            dirty_sources.append("kernel-builder")

        logger.warning(
            f"Kernel '{metadata.name}' variant '{variant}' was built from a dirty "
            f"git tree ({', '.join(dirty_sources)} had uncommitted changes). Its "
            "recorded git revision does not fully identify the sources it was built "
            "from, so the build may not be reproducible.",
            stacklevel=3,
        )


def _installed_version() -> Version | None:
    """The installed `kernels` version as a numeric version.

    Pre-release and development suffixes are stripped, since kernel metadata
    only records release versions. Without stripping, a development version
    like `0.17.0.dev0` would compare as older than the `0.17.0` it implements.

    Returns `None` if the installed version is not a PEP 440 version, which
    can happen for versions derived from a VCS checkout. Such a version cannot
    be compared, and this check must never make a kernel fail to load.
    """
    # Avoid an import cycle.
    from kernels import __version__

    try:
        release = parse(__version__).release
    except InvalidVersion:
        return None

    # packaging < 22 returns a LegacyVersion with `release = None` for
    # non-PEP 440 versions instead of raising `InvalidVersion`.
    if release is None:
        return None

    return Version.from_str(".".join(str(part) for part in release))


class MinverValidator:
    """Validate that the installed `kernels` library meets the minimum version
    required by a kernel."""

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        minver = metadata.kernels_minver
        if minver is None:
            return

        installed = _installed_version()
        if installed is not None and installed < minver:
            # Report the verbatim installed version, not the normalized one used
            # for comparison, so the message matches what `pip show` reports.
            # Avoid an import cycle.
            from kernels import __version__

            raise RuntimeError(
                f"Kernel '{metadata.name}' variant '{variant}' requires "
                f"kernels>={minver}, but version {__version__} is installed. "
                "Upgrade with: pip install --upgrade kernels"
            )


@dataclass
class AllValidator:
    """Apply multiple validators to a kernel dependency tree."""

    validators: list[MetadataValidator]

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        for validator in self.validators:
            validator.validate_metadata(metadata=metadata, variant=variant)


def default_metadata_validators() -> list[MetadataValidator]:
    """The metadata validators that are applied to every kernel dependency tree."""
    return [DependencyValidator(), MinverValidator(), DirtyValidator()]


# Kernel validators.


class KernelValidator(Protocol):
    """Kernel (build variant) validator."""

    def validate_kernel(self, *, kernel: "LocalKernel") -> None: ...


@dataclass
class SignatureValidator:
    policy: "VerificationPolicy | None" = None

    def validate_kernel(self, *, kernel: "LocalKernel") -> None:
        if not has_sigstore:
            return

        from .verify_cache import load_receipt, receipt_key

        if os.environ.get("KERNELS_VERIFY") == "always":
            self._verify(kernel)
            return

        key = receipt_key(kernel)
        receipt = load_receipt(key)
        if receipt is not None and self._verify_receipt(receipt, kernel):
            return

        self._verify(kernel, key=key)

    def _verify_receipt(self, receipt: "VerificationReceipt", kernel: "LocalKernel") -> bool:
        """Verify against a cached receipt. Returns `True` when a verdict was reached."""
        from sigstore.errors import VerificationError

        from .verify import DEFAULT_POLICY

        if not self._receipt_matches(receipt, kernel):
            logger.warning(f"Verification receipt does not match kernel, re-verifying: {kernel.variant_path}")
            return False

        # The kernel content itself is not re-validated: verification already
        # established that the metadata was correctly signed and that the
        # kernel hashes to the signed digest. Only the policy can have
        # changed since, so re-evaluate it against the stored certificate.
        policy = self.policy if self.policy is not None else DEFAULT_POLICY
        try:
            policy.verify(receipt.certificate)
        except VerificationError as e:
            # A full verification would evaluate the same certificate against
            # this policy and fail as well.
            logger.warning(f"Metadata signature verification failed:\n{e}: {kernel.variant_path}")
            return True

        logger.info(f"Kernel already verified: {kernel.variant_path}")
        return True

    @staticmethod
    def _receipt_matches(receipt: "VerificationReceipt", kernel: "LocalKernel") -> bool:
        """The receipt describes the kernel it is used for.

        Guards against receipts that were transplanted between receipt keys."""
        if receipt.variant != kernel.variant_str:
            return False
        if kernel.origin is not None:
            return receipt.repo_id == kernel.origin.repo_id and receipt.revision == kernel.origin.revision
        return receipt.path == str(kernel.variant_path.resolve())

    def _verify(self, kernel: "LocalKernel", key: str | None = None) -> None:
        from .verify import VerificationResult, verify_variant

        result = verify_variant(kernel.variant_path, policy=self.policy)
        match result:
            case VerificationResult.Success():
                logger.info(f"Kernel successfully verified: {kernel.variant_path}")
                if key is not None:
                    self._store_receipt(key, kernel)
            case _:
                message = str(result)
                logger.warning(f"{message[:1].upper()}{message[1:]}: {kernel.variant_path}")

    def _store_receipt(self, key: str, kernel: "LocalKernel") -> None:
        """Store a verification receipt. Failure degrades to re-verification on the next load."""
        from sigstore.models import Bundle

        from .verify_cache import VerificationReceipt, store_receipt

        try:
            bundle = Bundle.from_json((kernel.variant_path / "metadata.json.sigstore").read_bytes())
            metadata = Metadata.from_bytes((kernel.variant_path / "metadata.json").read_bytes())
            store_receipt(
                key,
                VerificationReceipt(
                    certificate=bundle.signing_certificate,
                    variant=kernel.variant_str,
                    digest=repr(metadata.digest),
                    verified_at=datetime.now(timezone.utc),
                    repo_id=kernel.origin.repo_id if kernel.origin is not None else None,
                    revision=kernel.origin.revision if kernel.origin is not None else None,
                    path=None if kernel.origin is not None else str(kernel.variant_path.resolve()),
                ),
            )
        except OSError as e:
            logger.debug(f"Cannot store verification receipt ({kernel.variant_path}): {e}")


def default_kernel_validators() -> list[KernelValidator]:
    """The kernel validators that are applied to every kernel dependency tree."""
    return [SignatureValidator()]


@dataclass
class AllKernelValidator:
    """Apply multiple validators to a kernel dependency tree."""

    validators: list[KernelValidator]

    def validate_kernel(self, *, kernel: "LocalKernel") -> None:
        for validator in self.validators:
            validator.validate_kernel(kernel=kernel)


if TYPE_CHECKING:
    # Ensure all validators obey the protocol.
    _metadata_validators: tuple[MetadataValidator, ...] = (
        DependencyValidator(),
        ArchValidator(),
        DirtyValidator(),
        MinverValidator(),
        AllValidator([]),
    )
    _kernel_validator: tuple[KernelValidator, ...] = (SignatureValidator(), AllKernelValidator([]))
