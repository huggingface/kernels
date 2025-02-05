import ctypes
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from importlib.metadata import Distribution
from types import ModuleType
from typing import List, Optional

from huggingface_hub import hf_hub_download, snapshot_download
from packaging.version import parse

from hf_kernels.compat import tomllib
from hf_kernels.lockfile import KernelLock

CACHE_DIR: Optional[str] = os.environ.get("HF_KERNELS_CACHE", None)


def build_variant():
    import torch

    torch_version = parse(torch.__version__)
    cuda_version = parse(torch.version.cuda)
    cxxabi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    cpu = platform.machine()
    os = platform.system().lower()

    return f"torch{torch_version.major}{torch_version.minor}-{cxxabi}-cu{cuda_version.major}{cuda_version.minor}-{cpu}-{os}"


def import_from_path(module_name: str, file_path):
    # We cannot use the module name as-is, after adding it to `sys.modules`,
    # it would also be used for other imports. So, we make a module name that
    # depends on the path for it to be unique using the hex-encoded hash of
    # the path.
    path_hash = "{:x}".format(ctypes.c_size_t(hash(file_path)).value)
    module_name = f"{module_name}_{path_hash}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_kernel(repo_id: str, revision: str, local_files_only: bool = False):
    package_name = get_metadata(repo_id, revision, local_files_only=local_files_only)[
        "torch"
    ]["name"]
    repo_path = snapshot_download(
        repo_id,
        allow_patterns=f"build/{build_variant()}/*",
        cache_dir=CACHE_DIR,
        revision=revision,
        local_files_only=local_files_only,
    )
    return package_name, f"{repo_path}/build/{build_variant()}"


def install_kernel_all_variants(
    repo_id: str, revision: str, local_files_only: bool = False
):
    snapshot_download(
        repo_id,
        allow_patterns="build/*",
        cache_dir=CACHE_DIR,
        revision=revision,
        local_files_only=local_files_only,
    )


def get_metadata(repo_id: str, revision: str, local_files_only: bool = False):
    with open(
        hf_hub_download(
            repo_id,
            "build.toml",
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=local_files_only,
        ),
        "rb",
    ) as f:
        return tomllib.load(f)


def get_kernel(repo_id: str, revision: str = "main"):
    package_name, package_path = install_kernel(repo_id, revision=revision)
    return import_from_path(package_name, f"{package_path}/{package_name}/__init__.py")


def load_kernel(repo_id: str):
    """Get a pre-downloaded, locked kernel."""
    locked_sha = _get_caller_locked_kernel(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    filename = hf_hub_download(
        repo_id,
        "build.toml",
        cache_dir=CACHE_DIR,
        local_files_only=True,
        revision=locked_sha,
    )
    with open(filename, "rb") as f:
        metadata = tomllib.load(f)
    package_name = metadata["torch"]["name"]

    repo_path = os.path.dirname(filename)
    package_path = f"{repo_path}/build/{build_variant()}"
    return import_from_path(package_name, f"{package_path}/{package_name}/__init__.py")


def get_locked_kernel(repo_id: str, local_files_only: bool = False):
    """Get a kernel using a lock file."""
    locked_sha = _get_caller_locked_kernel(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    package_name, package_path = install_kernel(
        repo_id, locked_sha, local_files_only=local_files_only
    )

    return import_from_path(package_name, f"{package_path}/{package_name}/__init__.py")


def _get_caller_locked_kernel(repo_id: str) -> Optional[str]:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("hf-kernels.lock")
        if lock_json is not None:
            for kernel_lock_json in json.loads(lock_json):
                kernel_lock = KernelLock.from_json(kernel_lock_json)
                if kernel_lock.repo_id == repo_id:
                    return kernel_lock.sha
    return None


def _get_caller_distributions() -> List[Distribution]:
    module = _get_caller_module()
    if module is None:
        return []

    # Look up all possible distributions that this module could be from.
    package = module.__name__.split(".")[0]
    dist_names = importlib.metadata.packages_distributions().get(package)
    if dist_names is None:
        return []

    return [importlib.metadata.distribution(dist_name) for dist_name in dist_names]


def _get_caller_module() -> Optional[ModuleType]:
    stack = inspect.stack()
    # Get first module in the stack that is not the current module.
    first_module = inspect.getmodule(stack[0][0])
    for frame in stack[1:]:
        module = inspect.getmodule(frame[0])
        if module is not None and module != first_module:
            return module
    return first_module
