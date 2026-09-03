import logging
from dataclasses import dataclass
from typing import Protocol

from kernels_data import Metadata, Version
from packaging.version import InvalidVersion, parse

from kernels.archs import _check_arch_incompatibility
from kernels.backends import _backend
from kernels.python_deps import validate_dependencies

logger = logging.getLogger(__name__)


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
class AllValidator:
    """Apply multiple validators to a kernel dependency tree."""

    validators: list[MetadataValidator]

    def validate_metadata(self, *, metadata: Metadata, variant: str) -> None:
        for validator in self.validators:
            validator.validate_metadata(metadata=metadata, variant=variant)


def default_metadata_validators() -> list[MetadataValidator]:
    """The metadata validators that are applied to every kernel dependency tree."""
    return [DependencyValidator(), MinverValidator(), DirtyValidator()]
