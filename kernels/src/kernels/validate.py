import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from packaging.version import InvalidVersion, parse

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

from kernels._rust import KernelLocation, Metadata, Version
from kernels.archs import _check_arch_incompatibility
from kernels.backends import _backend
from kernels.compat import has_sigstore
from kernels.python_deps import validate_dependencies
from kernels.resolver import LocalKernel

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sigstore.verify.policy import VerificationPolicy


# Metadata validators.


class MetadataValidator(Protocol):
    """Metadata validator for a kernel build variant."""

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
class AllMetadataValidator:
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
    """Verify the signature of a kernel build variant.

    Only kernels with a known Hub origin are verified, since local kernels
    are typically for development and not signed.

    Verification issues are currently reported as warnings. However, an
    exception will be raised in future versions."""

    policy: "VerificationPolicy | None" = None

    def validate_kernel(self, *, kernel: "LocalKernel") -> None:
        if not has_sigstore:
            return

        if kernel.origin is None:
            return

        # sigstore is still an optional dependency, so import lazily.
        from kernels.verify import VerificationResult, verify_variant

        location = KernelLocation.remote(
            kernel.origin.repo_id,
            kernel.origin.revision,
            kernel.variant_str,
        )

        result = verify_variant(kernel.variant_path, policy=self.policy, location=location)

        kernel_str = f"Kernel '{kernel.metadata.name}' variant '{kernel.variant_str}'"

        match result:
            case VerificationResult.Success():
                logger.debug(f"{kernel_str}: {result}")
            case VerificationResult.Failure():
                logger.warning(f"{kernel_str}: {result}", stacklevel=3)
            case _ as unreachable:
                assert_never(unreachable)


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


# Prototype type checks.

if TYPE_CHECKING:
    # Ensure all validators obey the protocol.
    _metadata_validators: tuple[MetadataValidator, ...] = (
        DependencyValidator(),
        ArchValidator(),
        DirtyValidator(),
        MinverValidator(),
        AllMetadataValidator([]),
    )
    _kernel_validator: tuple[KernelValidator, ...] = (SignatureValidator(), AllKernelValidator([]))
