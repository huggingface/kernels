import importlib.metadata
import inspect
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType

from huggingface_hub.hf_api import HfApi
from kernels_data import (
    KernelDependency,
    KernelLock,
    KernelLocks,
    Metadata,
)

from kernels._versions import resolve_kernel_version
from kernels.compat import tomllib
from kernels.hf_hub import CACHE_DIR, _check_trust_remote_code
from kernels.variants import get_variants


def dependency_locks(
    kernel: KernelDependency,
    *,
    api: HfApi,
    backend: str | None,
    seen: set[KernelDependency] | None = None,
    kernel_locks: dict[KernelDependency, KernelLock] | None = None,
) -> dict[KernelDependency, KernelLock]:

    if kernel_locks is None:
        kernel_locks = dict()

    # If the kernel is already in the locks, we have already resolved
    # the kernel and its dependencies.
    if kernel in kernel_locks:
        return kernel_locks

    if seen is None:
        seen = set()

    # Check for cycles.
    if kernel in seen:
        raise ValueError(f"Cyclic kernel dependency detected: {kernel.repo_id}")
    seen.add(kernel)

    # Check if the repo is trusted before downloading anything.
    _check_trust_remote_code(
        repo_id=kernel.repo_id,
        local_files_only=False,
        trust_remote_code=False,
    )

    revision = resolve_kernel_version(kernel, local_files_only=False)

    for variant in get_variants(api, repo_id=kernel.repo_id, revision=revision):
        metadata_path = Path(
            api.hf_hub_download(
                kernel.repo_id,
                repo_type="kernel",
                filename=f"build/{variant.variant_str}/metadata.json",
                cache_dir=CACHE_DIR,
                revision=revision,
                local_files_only=False,
            )
        )
        metadata = Metadata.read_from_file(metadata_path)

        kernel_deps = {}
        for dep in metadata.kernel_depends:
            kernel_deps[dep] = dependency_locks(dep, api=api, backend=backend, seen=seen, kernel_locks=kernel_locks)

    seen.remove(kernel)

    kernel_locks[kernel] = KernelLock(
        commit=revision,
    )

    return kernel_locks


def extract_dependency_locks(
    kernels: list[KernelDependency],
    *,
    api: HfApi,
    backend: str | None,
) -> KernelLocks:
    kernel_locks = None
    for kernel in kernels:
        kernel_locks = dependency_locks(kernel, api=api, backend=backend, kernel_locks=kernel_locks)

    if kernel_locks is None:
        raise ValueError("No kernels found to lock.")

    return KernelLocks(locks=kernel_locks)


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


def _get_locked_kernel_revision(repo_id: str, lock_json: str) -> tuple[KernelLocks, KernelDependency]:
    kernel_locks = KernelLocks.from_json(lock_json)

    # Lock files are keyed `KernelDependency`, but for project-locked
    # kenels we only have one version at the top level, so we have to
    # do a linear search.
    kernel_dep = next((dep for dep in kernel_locks.keys() if dep.repo_id == repo_id), None)

    if kernel_dep is None:
        raise ValueError(
            f"Kernel `{repo_id}` is not locked. Please lock it with `kernels lock <project>` and then reinstall the project."
        )

    return kernel_locks, kernel_dep


def get_locked_kernel_revision(repo_id: str, lockfile: Path) -> tuple[KernelLocks, KernelDependency]:
    with open(lockfile, "r") as f:
        return _get_locked_kernel_revision(repo_id, f.read())


def get_caller_locked_kernel_revision(
    repo_id: str,
) -> tuple[KernelLocks, KernelDependency]:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue

        return _get_locked_kernel_revision(repo_id, lock_json)

    raise ValueError(
        "Could not find a `kernels.lock` file in the caller's package metadata. Please lock kernels with `kernels lock <project>` and then reinstall the project."
    )


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
