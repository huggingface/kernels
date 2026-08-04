import hashlib
import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType

from huggingface_hub.dataclasses import strict
from huggingface_hub.hf_api import RepoFile

from kernels._versions import resolve_version_spec_as_ref
from kernels.compat import tomllib
from kernels.status import resolve_status


@strict
@dataclass
class VariantLock:
    hash: str
    hash_type: str = "git_lfs_concat"


@strict
@dataclass
class KernelLock:
    repo_id: str
    sha: str
    variants: dict[str, VariantLock]

    @classmethod
    def from_json(cls, o: dict):
        variants = {variant: VariantLock(**lock) for variant, lock in o["variants"].items()}
        return cls(repo_id=o["repo_id"], sha=o["sha"], variants=variants)


def get_kernel_locks(repo_id: str, version_spec: int) -> KernelLock:
    """
    Get the locks for a kernel with the given version.
    """
    from kernels.utils import _get_hf_api

    api = _get_hf_api()

    # NOTE: the destination of a redirect is respected but we still use
    # resolve_version_spec_as_ref to resolve the version specifier of the
    # final destination repo.
    repo_id, _ = resolve_status(api, repo_id, "main")

    tag_for_newest = resolve_version_spec_as_ref(repo_id, version_spec)

    revision = tag_for_newest.target_commit

    r = api.repo_info(
        repo_id=repo_id,
        repo_type="kernel",
        revision=revision,
    )
    if r.sha is None:
        raise ValueError(f"Cannot get commit SHA for repo {repo_id} for tag {tag_for_newest.name}")

    siblings = [
        f
        for f in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="kernel",
            revision=revision,
            recursive=True,
        )
        if isinstance(f, RepoFile)
    ]

    variant_files: dict[str, list[tuple[bytes, str]]] = {}
    for sibling in siblings:
        if sibling.rfilename.startswith("build/torch"):
            if sibling.blob_id is None:
                raise ValueError(f"Cannot get blob ID for {sibling.rfilename}")

            # Exclude Python bytecode. If bytecode exists, it is generated
            # by the interpreter, since we exclude bytecode from builds
            # and downloads.
            if sibling.rfilename.endswith(".pyc") or "__pycache__" in sibling.rfilename.split("/"):
                continue

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


def get_locked_kernel_revision(repo_id: str, lock_json: str) -> str | None:
    for kernel_lock_json in json.loads(lock_json):
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        if kernel_lock.repo_id == repo_id:
            return kernel_lock.sha
    return None


def get_caller_locked_kernel_revision(repo_id: str) -> str | None:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue
        locked_sha = get_locked_kernel_revision(repo_id, lock_json)
        if locked_sha is not None:
            return locked_sha
    return None


def _get_caller_distributions() -> list[Distribution]:
    module = _get_caller_module()
    if module is None:
        return []

    # Look up all possible distributions that this module could be from.
    package = module.__name__.split(".")[0]
    dist_names = importlib.metadata.packages_distributions().get(package)
    if dist_names is None:
        return []

    return [importlib.metadata.distribution(dist_name) for dist_name in dist_names]


def _get_caller_module() -> ModuleType | None:
    stack = inspect.stack()
    # Get first module in the stack that is not the current module.
    first_module = inspect.getmodule(stack[0][0])
    for frame in stack[1:]:
        module = inspect.getmodule(frame[0])
        if module is not None and module != first_module:
            return module
    return first_module
