from pathlib import Path
from typing import Any, Dict, List
import json
import argparse
import dataclasses
from dataclasses import dataclass
import tomllib

from huggingface_hub import HfApi


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


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


def lock_kernels():
    parser = argparse.ArgumentParser(
        prog="hf-lock-kernels", description="Lock kernel revisions"
    )
    parser.add_argument(
        "project_dir",
        type=Path,
        help="The project directory",
    )
    args = parser.parse_args()

    with open(args.project_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = _get_nested_attr(data, ["tool", "kernels", "dependencies"])

    all_locks = []
    for kernel, version in kernel_versions.items():
        all_locks.append(get_kernel_locks(kernel, version))

    with open(args.project_dir / "kernels.lock", "w") as f:
        json.dump(all_locks, f, cls=_JSONEncoder, indent=2)


def get_kernel_locks(repo_id: str, revision: str):
    r = HfApi().repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    if r.sha is None:
        raise ValueError(
            f"Cannot get commit SHA for repo {repo_id} at revision {revision}"
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


def _get_nested_attr(d, attr: List[str]) -> Any:
    for a in attr:
        d = d.get(a)
        if d is None:
            break
    return d


def write_egg_lockfile(cmd, basename, filename):
    import logging

    cwd = Path.cwd()
    with open(cwd / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    kernel_versions = _get_nested_attr(data, ["tool", "kernels", "dependencies"])
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
