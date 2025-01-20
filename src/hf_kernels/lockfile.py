from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi

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


def get_kernel_locks(repo_id: str, revision: str):
    r = HfApi().repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if r.sha is None:
        raise ValueError(
            f"Cannot get commit SHA for repo {repo_id} at revision {revision}"
        )

    if r.siblings is None:
        raise ValueError(
            f"Cannot get sibling information for {repo_id} at revision {revision}"
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
    with open(cwd / "pyproject.toml", "rb") as f:
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
