import importlib.metadata
import inspect
import json
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType

from huggingface_hub.dataclasses import strict
from kernels_data import KernelDependency, KernelVersion

from kernels.compat import tomllib
from kernels.deps import DepTreeNode, LocalKernel, RemoteKernel, resolve_kernel_tree


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
    kernel_depends: dict[str, "KernelLock"]

    @classmethod
    def from_json(cls, o: dict):
        kernel_depends = o.get("kernel_depends", {})
        kernel_depends = {dep_repo_id: cls.from_json(lock_json) for dep_repo_id, lock_json in kernel_depends.items()}
        return cls(
            repo_id=o["repo_id"],
            sha=o["sha"],
            kernel_depends=kernel_depends,
        )


def extract_locks(tree: DepTreeNode[LocalKernel | RemoteKernel]) -> KernelLock:
    if isinstance(tree.location, LocalKernel):
        raise ValueError("Cannot extract locks from a local kernel")

    repo_id = tree.location.repo_id
    sha = tree.location.revision

    locked_deps = {}
    for name, dep_tree in tree.depends.items():
        locked_deps[name] = extract_locks(dep_tree)

    return KernelLock(repo_id=repo_id, sha=sha, kernel_depends=locked_deps)


def get_kernel_locks(repo_id: str, version_spec: int) -> KernelLock:
    """
    Get the locks for a kernel with the given version.
    """
    from kernels.hf_hub import _get_hf_api

    api = _get_hf_api()

    tree = resolve_kernel_tree(
        api=api,
        backend=None,
        local_kernels={},
        kernel=KernelDependency(repo_id=repo_id, version=KernelVersion.Version(version_spec)),
        local_files_only=False,
        kernel_locks=None,
        # TODO: what is the right policy here?
        trust_remote_code=False,
    )

    return extract_locks(tree)


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


def get_locked_kernel_revisions(lock_json: str) -> dict[str, str]:
    kernel_locks = {}

    for kernel_lock_json in json.loads(lock_json):
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        kernel_locks[kernel_lock.repo_id] = kernel_lock.sha

    return kernel_locks


def get_caller_locked_kernel_revisions() -> dict[str, str]:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue
        kernel_locks = get_locked_kernel_revisions(lock_json)
        if len(kernel_locks) > 0:
            return kernel_locks
    return {}


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
