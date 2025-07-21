from typing import Dict

from huggingface_hub import HfApi
from huggingface_hub.hf_api import GitRefInfo
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


def _get_available_versions(repo_id: str) -> Dict[Version, GitRefInfo]:
    """Get kernel versions that are available in the repository."""
    versions = {}
    for tag in HfApi().list_repo_refs(repo_id).tags:
        if not tag.name.startswith("v"):
            continue
        try:
            versions[Version(tag.name[1:])] = tag
        except InvalidVersion:
            continue

    return versions


def resolve_version_spec_as_ref(repo_id: str, version_spec: str) -> GitRefInfo:
    """
    Get the locks for a kernel with the given version spec.

    The version specifier can be any valid Python version specifier:
    https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers
    """
    versions = _get_available_versions(repo_id)
    requirement = SpecifierSet(version_spec)
    accepted_versions = sorted(requirement.filter(versions.keys()))

    if len(accepted_versions) == 0:
        raise ValueError(
            f"No version of `{repo_id}` satisfies requirement: {version_spec}"
        )

    return versions[accepted_versions[-1]]
