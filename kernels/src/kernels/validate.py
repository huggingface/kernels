from dataclasses import dataclass
from typing import Protocol

from kernels_data import Version
from packaging.version import InvalidVersion, parse

from kernels.archs import _check_arch_incompatibility
from kernels.backends import _backend
from kernels.deps import DepTreeNode
from kernels.python_deps import validate_dependencies
from kernels.resolver import LocalKernel, RemoteKernel


class Validator(Protocol):
    """Validator for a node in a resolved kernel dependency tree."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None: ...


class DependencyValidator:
    """Validate the Python dependencies of a kernel tree node."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        validate_dependencies(
            tree.location.metadata.name.python_name,
            tree.location.metadata.python_depends,
            _backend(),
        )


class ArchValidator:
    """Validate architecture compatibility for a kernel tree node."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        _check_arch_incompatibility(tree.location.metadata, tree.location.variant_str)


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

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        metadata = tree.location.metadata
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
                f"Kernel '{metadata.name}' variant '{tree.location.variant_str}' requires "
                f"kernels>={minver}, but version {__version__} is installed. "
                "Upgrade with: pip install --upgrade kernels"
            )


@dataclass
class AllValidator:
    """Apply multiple validators to a kernel dependency tree."""

    validators: list[Validator]

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        for validator in self.validators:
            validator.validate(tree=tree)


def default_validators() -> list[Validator]:
    """The validators that are applied to every kernel dependency tree."""
    return [DependencyValidator(), MinverValidator()]
