import warnings

from huggingface_hub import HfApi
from huggingface_hub.hf_api import GitRefInfo
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


def _get_available_versions(repo_id: str) -> dict[int, GitRefInfo]:
    """Get kernel versions that are available in the repository."""
    versions = {}
    for branch in HfApi().list_repo_refs(repo_id).branches:
        if not branch.name.startswith("v"):
            continue
        try:
            versions[int(branch.name[1:])] = branch
        except ValueError:
            continue

    return versions


def _get_available_versions_old(repo_id: str) -> dict[Version, GitRefInfo]:
    """
    Get kernel versions that are available in the repository.

    This is for the old tag-based versioning scheme.
    """
    versions = {}
    for tag in HfApi().list_repo_refs(repo_id).tags:
        if not tag.name.startswith("v"):
            continue
        try:
            versions[Version(tag.name[1:])] = tag
        except InvalidVersion:
            continue

    return versions


def resolve_version_spec_as_ref(repo_id: str, version_spec: int | str) -> GitRefInfo:
    """
    Get the locks for a kernel with the given version spec.

    The version specifier can be any valid Python version specifier:
    https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
    """
    if isinstance(version_spec, int):
        versions = _get_available_versions(repo_id)
        ref = versions.get(version_spec, None)
        if ref is None:
            raise ValueError(
                f"Version {version_spec} not found, available versions: {', '.join(sorted(str(v) for v in versions.keys()))}"
            )
        return ref
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

        return versions_old[accepted_versions[-1]]


def select_revision_or_version(
    repo_id: str,
    *,
    revision: str | None,
    version: int | str | None,
) -> str:
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")

    if revision is not None:
        return revision
    elif version is not None:
        return resolve_version_spec_as_ref(repo_id, version).target_commit

    # Re-enable once we have proper UX on the hub for showing the
    # kernel versions.
    #
    # warnings.warn(
    #    "Future versions of `kernels` (>=0.14) will require specifying a kernel version or revision."
    #    "See: https://huggingface.co/docs/kernels/migration"
    # )

    return "main"
