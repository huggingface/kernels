from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from hf_kernels.compat import tomllib


@dataclass
class FileLock:
    filename: str
    blob_id: str


@dataclass
class KernelLock:
    repo_id: str
    sha: str
    files: List[FileLock]

    @classmethod
    def from_json(cls, o: Dict):
        files = [FileLock(**f) for f in o["files"]]
        return cls(repo_id=o["repo_id"], sha=o["sha"], files=files)


def _get_available_versions(repo_id: str):
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


def get_kernel_locks(repo_id: str, version_spec: str):
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

    tag_for_newest = versions[accepted_versions[-1]]

    r = HfApi().repo_info(
        repo_id=repo_id, revision=tag_for_newest.target_commit, files_metadata=True
    )
    if r.sha is None:
        raise ValueError(
            f"Cannot get commit SHA for repo {repo_id} for tag {tag_for_newest.name}"
        )

    if r.siblings is None:
        raise ValueError(
            f"Cannot get sibling information for {repo_id} for tag {tag_for_newest.name}"
        )

    file_locks = []
    for sibling in r.siblings:
        if sibling.rfilename.startswith("build/torch"):
            if sibling.blob_id is None:
                raise ValueError(f"Cannot get blob ID for {sibling.rfilename}")

            file_locks.append(
                FileLock(filename=sibling.rfilename, blob_id=sibling.blob_id)
            )

    return KernelLock(repo_id=repo_id, sha=r.sha, files=file_locks)


def write_egg_lockfile(cmd, basename, filename):
    import logging

    cwd = Path.cwd()
    pyproject_path = cwd / "pyproject.toml"
    if not pyproject_path.exists():
        # Nothing to do if the project doesn't have pyproject.toml.
        return

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    kernel_versions = data.get("tool", {}).get("kernels", {}).get("dependencies", None)
    if kernel_versions is None:
        return

    lock_path = cwd / "hf-kernels.lock"
    if not lock_path.exists():
        logging.warning(f"Lock file {lock_path} does not exist")
        # Ensure that the file gets deleted in editable installs.
        data = None
    else:
        data = open(lock_path, "r").read()

    cmd.write_or_delete_file(basename, filename, data)
