import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from huggingface_hub import HfApi
from huggingface_hub.hf_api import GitRefInfo
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from kernels.compat import tomllib


@dataclass
class VariantLock:
    hash: str
    hash_type: str = "git_lfs_concat"


@dataclass
class KernelLock:
    repo_id: str
    sha: str
    variants: Dict[str, VariantLock]

    @classmethod
    def from_json(cls, o: Dict):
        variants = {
            variant: VariantLock(**lock) for variant, lock in o["variants"].items()
        }
        return cls(repo_id=o["repo_id"], sha=o["sha"], variants=variants)


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


def get_kernel_locks(repo_id: str, version_spec: str) -> KernelLock:
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

    variant_files: Dict[str, List[Tuple[bytes, str]]] = {}
    for sibling in r.siblings:
        if sibling.rfilename.startswith("build/torch"):
            if sibling.blob_id is None:
                raise ValueError(f"Cannot get blob ID for {sibling.rfilename}")

            path = Path(sibling.rfilename)
            variant = path.parts[1]
            filename = Path(*path.parts[2:])

            hash = sibling.lfs.sha256 if sibling.lfs is not None else sibling.blob_id

            files = variant_files.setdefault(variant, [])

            # Encode as posix for consistent slash handling, then encode
            # as utf-8 for byte-wise sorting later.
            files.append((filename.as_posix().encode("utf-8"), hash))

    variant_locks = {}
    for variant, files in variant_files.items():
        m = hashlib.sha256()
        for filename_bytes, hash in sorted(files):
            # Filename as bytes.
            m.update(filename_bytes)
            # Git blob or LFS file hash as bytes.
            m.update(bytes.fromhex(hash))

        variant_locks[variant] = VariantLock(hash=f"sha256-{m.hexdigest()}")

    return KernelLock(repo_id=repo_id, sha=r.sha, variants=variant_locks)


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

    lock_path = cwd / "kernels.lock"
    if not lock_path.exists():
        logging.warning(f"Lock file {lock_path} does not exist")
        # Ensure that the file gets deleted in editable installs.
        data = None
    else:
        data = open(lock_path, "r").read()

    cmd.write_or_delete_file(basename, filename, data)
