import json
import warnings
from pathlib import Path

from kernels_data import Metadata, Version
from packaging.version import InvalidVersion, parse


def _read_kernels_minver(variant_path: Path) -> Version | None:
    """The minimum `kernels` version required by a build variant."""

    # Note, for kernels >= 0.17, we read the version from the metadata,
    # but to minimize the impact in the backport, we are reading from
    # metadata.json here directly.

    try:
        with open(variant_path / "metadata.json", "rb") as f:
            minver = json.load(f).get("kernels-minver")
    except (OSError, ValueError):
        return None

    if not isinstance(minver, str):
        return None

    try:
        return Version.from_str(minver)
    except ValueError:
        return None


def _installed_version() -> Version | None:
    """The installed `kernels` version as a numeric version."""
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


def _warn_if_below_minver(metadata: Metadata, variant_path: Path) -> None:
    """Warn when the installed `kernels` library is older than the minimum
    version required by a kernel.
    """
    minver = _read_kernels_minver(variant_path)
    if minver is None:
        return

    installed = _installed_version()
    if installed is not None and installed < minver:
        # Report the verbatim installed version, not the normalized one used
        # for comparison, so the message matches what `pip show` reports.
        from kernels import __version__

        warnings.warn(
            f"Kernel '{metadata.name}' variant '{variant_path.name}' requires "
            f"kernels>={minver}, but version {__version__} is installed. "
            "The kernel may not load or work correctly; upgrade with: "
            "pip install --upgrade kernels",
            stacklevel=3,
        )
