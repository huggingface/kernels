import logging
import warnings

from huggingface_hub.hf_api import GitRefInfo
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)


def _get_available_versions(
    repo_id: str,
) -> tuple[dict[int, GitRefInfo], str]:
    """Get kernel versions that are available in the repository.

    Tries ``"kernel"`` repo type first. If the repository is not found,
    falls back to ``"model"``.

    Returns a tuple of (versions, repo_type)."""
    from kernels.utils import _get_hf_api, _resolve_repo_type

    repo_type = _resolve_repo_type(repo_id)

    refs = _get_hf_api().list_repo_refs(repo_id=repo_id, repo_type=repo_type)

    versions = {}
    for branch in refs.branches:
        if not branch.name.startswith("v"):
            continue
        try:
            versions[int(branch.name[1:])] = branch
        except ValueError:
            continue

    return versions, repo_type


def _get_available_versions_old(repo_id: str) -> dict[Version, GitRefInfo]:
    """
    Get kernel versions that are available in the repository.

    This is for the old tag-based versioning scheme.
    """
    from kernels.utils import _get_hf_api, _resolve_repo_type

    repo_type = _resolve_repo_type(repo_id)
    versions = {}
    for tag in _get_hf_api().list_repo_refs(repo_id, repo_type=repo_type).tags:
        if not tag.name.startswith("v"):
            continue
        try:
            versions[Version(tag.name[1:])] = tag
        except InvalidVersion:
            continue

    return versions


def resolve_version_spec_as_ref(
    repo_id: str, version_spec: int | str
) -> tuple[GitRefInfo, str]:
    """
    Get the ref for a kernel with the given version spec.
    The version specifier can be any valid Python version specifier:
    https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers

    Returns a tuple of (ref, repo_type).
    """
    if isinstance(version_spec, int):
        versions, repo_type = _get_available_versions(repo_id)

        ref = versions.get(version_spec, None)
        if ref is None:
            raise ValueError(
                f"Version {version_spec} not found, available versions: {', '.join(sorted(str(v) for v in versions.keys()))}"
            )

        latest_version = max(versions.keys())
        if version_spec < latest_version:
            logger.warning(
                "You are using version %d of '%s', but version %d is available.",
                version_spec,
                repo_id,
                latest_version,
            )

        return ref, repo_type
    else:
        warnings.warn(
            """Version specifiers are deprecated, support will be removed in a future `kernels` version.
            For more information on migrating to versions, see: https://huggingface.co/docs/kernels/migration"""
        )
        versions_old = _get_available_versions_old(repo_id)
        requirement = SpecifierSet(version_spec)
        accepted_versions = sorted(requirement.filter(versions_old.keys()))

        if len(accepted_versions) == 0:
            raise ValueError(
                f"No version of `{repo_id}` satisfies requirement: {version_spec}"
            )

        return versions_old[accepted_versions[-1]], "model"


def select_revision_or_version(
    repo_id: str,
    *,
    revision: str | None,
    version: int | str | None,
) -> tuple[str, str | None]:
    """Select a revision, returning (revision, repo_type)."""
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")

    if revision is not None:
        return revision, None
    elif version is not None:
        ref, repo_type = resolve_version_spec_as_ref(repo_id, version)
        return ref.target_commit, repo_type

    # Re-enable once we have proper UX on the hub for showing the
    # kernel versions.
    #
    # warnings.warn(
    #    "Future versions of `kernels` (>=0.14) will require specifying a kernel version or revision."
    #    "See: https://huggingface.co/docs/kernels/migration"
    # )

    return "main", None
