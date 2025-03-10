import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download
from packaging.version import parse

from kernels.lockfile import KernelLock, VariantLock

CACHE_DIR: Optional[str] = os.environ.get("HF_KERNELS_CACHE", None)


def build_variant() -> str:
    import torch

    if torch.version.cuda is None:
        raise AssertionError(
            "This kernel requires CUDA to be installed. Torch was not compiled with CUDA enabled."
        )

    torch_version = parse(torch.__version__)
    cuda_version = parse(torch.version.cuda)
    cxxabi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    cpu = platform.machine()
    os = platform.system().lower()

    return f"torch{torch_version.major}{torch_version.minor}-{cxxabi}-cu{cuda_version.major}{cuda_version.minor}-{cpu}-{os}"


def universal_build_variant() -> str:
    # Once we support other frameworks, detection goes here.
    return "torch-universal"


def import_from_path(module_name: str, file_path: Path) -> ModuleType:
    # We cannot use the module name as-is, after adding it to `sys.modules`,
    # it would also be used for other imports. So, we make a module name that
    # depends on the path for it to be unique using the hex-encoded hash of
    # the path.
    path_hash = "{:x}".format(ctypes.c_size_t(hash(file_path)).value)
    module_name = f"{module_name}_{path_hash}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Cannot load module {module_name} from spec")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def install_kernel(
    repo_id: str,
    revision: str,
    local_files_only: bool = False,
    variant_locks: Optional[Dict[str, VariantLock]] = None,
) -> Tuple[str, Path]:
    """
    Download a kernel for the current environment to the cache.

    The output path is validated againt `hash` when set.
    """
    package_name = package_name_from_repo_id(repo_id)
    variant = build_variant()
    universal_variant = universal_build_variant()
    repo_path = Path(
        snapshot_download(
            repo_id,
            allow_patterns=[f"build/{variant}/*", f"build/{universal_variant}/*"],
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=local_files_only,
        )
    )

    variant_path = repo_path / "build" / variant
    universal_variant_path = repo_path / "build" / universal_variant

    if not variant_path.exists() and universal_variant_path.exists():
        # Fall back to universal variant.
        variant = universal_variant
        variant_path = universal_variant_path

    if variant_locks is not None:
        variant_lock = variant_locks.get(variant)
        if variant_lock is None:
            raise ValueError(f"No lock found for build variant: {variant}")
        validate_kernel(repo_path=repo_path, variant=variant, hash=variant_lock.hash)

    module_init_path = variant_path / package_name / "__init__.py"

    if not os.path.exists(module_init_path):
        raise FileNotFoundError(
            f"Kernel `{repo_id}` at revision {revision} does not have build: {variant}"
        )

    return package_name, variant_path


def install_kernel_all_variants(
    repo_id: str,
    revision: str,
    local_files_only: bool = False,
    variant_locks: Optional[Dict[str, VariantLock]] = None,
) -> Path:
    repo_path = Path(
        snapshot_download(
            repo_id,
            allow_patterns="build/*",
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=local_files_only,
        )
    )

    if variant_locks is not None:
        for entry in (repo_path / "build").iterdir():
            variant = entry.parts[-1]

            variant_lock = variant_locks.get(variant)
            if variant_lock is None:
                raise ValueError(f"No lock found for build variant: {variant}")

            validate_kernel(
                repo_path=repo_path, variant=variant, hash=variant_lock.hash
            )

    return repo_path / "build"


def get_kernel(repo_id: str, revision: str = "main") -> ModuleType:
    package_name, package_path = install_kernel(repo_id, revision=revision)
    return import_from_path(package_name, package_path / package_name / "__init__.py")


def load_kernel(repo_id: str, *, lockfile: Optional[Path] = None) -> ModuleType:
    """
    Get a pre-downloaded, locked kernel.

    If `lockfile` is not specified, the lockfile will be loaded from the
    caller's package metadata.
    """
    if lockfile is None:
        locked_sha = _get_caller_locked_kernel(repo_id)
    else:
        with open(lockfile, "r") as f:
            locked_sha = _get_locked_kernel(repo_id, f.read())

    if locked_sha is None:
        raise ValueError(
            f"Kernel `{repo_id}` is not locked. Please lock it with `kernels lock <project>` and then reinstall the project."
        )

    package_name = package_name_from_repo_id(repo_id)

    variant = build_variant()
    universal_variant = universal_build_variant()

    repo_path = Path(
        snapshot_download(
            repo_id,
            allow_patterns=[f"build/{variant}/*", f"build/{universal_variant}/*"],
            cache_dir=CACHE_DIR,
            revision=locked_sha,
            local_files_only=True,
        )
    )

    variant_path = repo_path / "build" / variant
    universal_variant_path = repo_path / "build" / universal_variant
    if not variant_path.exists() and universal_variant_path.exists():
        # Fall back to universal variant.
        variant = universal_variant
        variant_path = universal_variant_path

    module_init_path = variant_path / package_name / "__init__.py"
    if not os.path.exists(module_init_path):
        raise FileNotFoundError(
            f"Locked kernel `{repo_id}` does not have build `{variant}` or was not downloaded with `kernels download <project>`"
        )

    return import_from_path(package_name, variant_path / package_name / "__init__.py")


def get_locked_kernel(repo_id: str, local_files_only: bool = False) -> ModuleType:
    """Get a kernel using a lock file."""
    locked_sha = _get_caller_locked_kernel(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    package_name, package_path = install_kernel(
        repo_id, locked_sha, local_files_only=local_files_only
    )

    return import_from_path(package_name, package_path / package_name / "__init__.py")


def _get_caller_locked_kernel(repo_id: str) -> Optional[str]:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue
        locked_sha = _get_locked_kernel(repo_id, lock_json)
        if locked_sha is not None:
            return locked_sha
    return None


def _get_locked_kernel(repo_id: str, lock_json: str) -> Optional[str]:
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


def validate_kernel(*, repo_path: Path, variant: str, hash: str):
    """Validate the given build variant of a kernel against a hasht."""
    variant_path = repo_path / "build" / variant

    # Get the file paths. The first element is a byte-encoded relative path
    # used for sorting. The second element is the absolute path.
    files: List[Tuple[bytes, Path]] = []
    # Ideally we'd use Path.walk, but it's only available in Python 3.12.
    for dirpath, _, filenames in os.walk(variant_path):
        for filename in filenames:
            file_abs = Path(dirpath) / filename

            # Python likes to create files when importing modules from the
            # cache, only hash files that are symlinked blobs.
            if file_abs.is_symlink():
                files.append(
                    (
                        file_abs.relative_to(variant_path).as_posix().encode("utf-8"),
                        file_abs,
                    )
                )

    m = hashlib.sha256()

    for filename_bytes, full_path in sorted(files):
        m.update(filename_bytes)

        blob_filename = full_path.resolve().name
        if len(blob_filename) == 40:
            # SHA-1 hashed, so a Git blob.
            m.update(git_hash_object(full_path.read_bytes()))
        elif len(blob_filename) == 64:
            # SHA-256 hashed, so a Git LFS blob.
            m.update(hashlib.sha256(full_path.read_bytes()).digest())
        else:
            raise ValueError(f"Unexpected blob filename length: {len(blob_filename)}")

    computedHash = f"sha256-{m.hexdigest()}"
    if computedHash != hash:
        raise ValueError(
            f"Lock file specifies kernel with hash {hash}, but downloaded kernel has hash: {computedHash}"
        )


def git_hash_object(data: bytes, object_type: str = "blob"):
    """Calculate git SHA1 of data."""
    header = f"{object_type} {len(data)}\0".encode()
    m = hashlib.sha1()
    m.update(header)
    m.update(data)
    return m.digest()


def package_name_from_repo_id(repo_id: str) -> str:
    return repo_id.split("/")[-1].replace("-", "_")
