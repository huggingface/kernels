import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import sys
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union

from huggingface_hub import file_exists, snapshot_download
from packaging.version import parse

from kernels._system import glibc_version
from kernels._versions import select_revision_or_version
from kernels.deps import validate_dependencies
from kernels.lockfile import KernelLock, VariantLock

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _get_cache_dir() -> Optional[str]:
    """Returns the kernels cache directory."""
    cache_dir = os.environ.get("HF_KERNELS_CACHE", None)
    if cache_dir is not None:
        logging.warning(
            "HF_KERNELS_CACHE will be removed in the future, use KERNELS_CACHE instead"
        )
        return cache_dir

    return os.environ.get("KERNELS_CACHE", None)


CACHE_DIR: Optional[str] = _get_cache_dir()


def _get_privateuse_backend_name() -> Optional[str]:
    import torch

    if hasattr(torch._C, "_get_privateuse1_backend_name"):
        return torch._C._get_privateuse1_backend_name()
    return None


def backend() -> str:
    import torch

    if torch.version.cuda is not None:
        return "cuda"
    elif torch.version.hip is not None:
        return "hip"
    elif torch.backends.mps.is_available():
        return "metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        return "xpu"
    elif _get_privateuse_backend_name() == "npu":
        return "cann"
    else:
        return "cpu"


def build_variant() -> str:
    import torch

    if torch.version.cuda is not None:
        cuda_version = parse(torch.version.cuda)
        compute_framework = f"cu{cuda_version.major}{cuda_version.minor}"
    elif torch.version.hip is not None:
        rocm_version = parse(torch.version.hip.split("-")[0])
        compute_framework = f"rocm{rocm_version.major}{rocm_version.minor}"
    elif torch.backends.mps.is_available():
        compute_framework = "metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        version = torch.version.xpu
        compute_framework = f"xpu{version[0:4]}{version[5:6]}"
    elif _get_privateuse_backend_name() == "npu":
        from torch_npu.utils.collect_env import get_cann_version  # type: ignore[import-not-found]

        cann_major, cann_minor = get_cann_version()[0], get_cann_version()[2]
        compute_framework = f"cann{cann_major}{cann_minor}"
    else:
        compute_framework = "cpu"

    torch_version = parse(torch.__version__)
    cpu = platform.machine()
    os = platform.system().lower()

    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
        return f"torch{torch_version.major}{torch_version.minor}-{compute_framework}-{cpu}-{os}"
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu
        return f"torch{torch_version.major}{torch_version.minor}-{compute_framework}-{cpu}-{os}"

    cxxabi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    return f"torch{torch_version.major}{torch_version.minor}-{cxxabi}-{compute_framework}-{cpu}-{os}"


def build_variant_noarch() -> str:
    import torch

    if torch.version.cuda is not None:
        return "torch-cuda"
    elif torch.version.hip is not None:
        return "torch-rocm"
    elif torch.backends.mps.is_available():
        return "torch-metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        return "torch-xpu"
    elif _get_privateuse_backend_name() == "npu":
        return "torch-npu"
    else:
        return "torch-cpu"


def build_variant_universal() -> str:
    # Once we support other frameworks, detection goes here.
    return "torch-universal"


def build_variants() -> List[str]:
    """Return compatible build variants in preferred order."""
    return [build_variant(), build_variant_noarch(), build_variant_universal()]


def _import_from_path(module_name: str, variant_path: Path) -> ModuleType:
    metadata_path = variant_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            deps = metadata.get("python-depends", [])
            validate_dependencies(deps, backend())

    file_path = variant_path / "__init__.py"
    if not file_path.exists():
        file_path = variant_path / module_name / "__init__.py"

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
    user_agent: Optional[Union[str, dict]] = None,
) -> Tuple[str, Path]:
    """
    Download a kernel for the current environment to the cache.

    The output path is validated against the hashes in `variant_locks` when provided.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`):
            The specific revision (branch, tag, or commit) to download.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only use local files and not download from the Hub.
        variant_locks (`Dict[str, VariantLock]`, *optional*):
            Optional dictionary of variant locks for validation.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.

    Returns:
        `Tuple[str, Path]`: A tuple containing the package name and the path to the variant directory.
    """
    package_name = package_name_from_repo_id(repo_id)
    allow_patterns = [f"build/{variant}/*" for variant in build_variants()]
    user_agent = _get_user_agent(user_agent=user_agent)
    repo_path = Path(
        snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=local_files_only,
            user_agent=user_agent,
        )
    )

    try:
        return _find_kernel_in_repo_path(repo_path, package_name, variant_locks)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot install kernel from repo {repo_id} (revision: {revision})"
        )


def _find_kernel_in_repo_path(
    repo_path: Path,
    package_name: str,
    variant_locks: Optional[Dict[str, VariantLock]] = None,
) -> Tuple[str, Path]:
    variants = build_variants()
    variant = None
    variant_path = None
    for candidate_variant in variants:
        variant_path = repo_path / "build" / candidate_variant
        if variant_path.exists():
            variant = candidate_variant
            break

    if variant is None:
        raise FileNotFoundError(
            f"Kernel at path `{repo_path}` does not have one of build variants: {', '.join(variants)}"
        )

    assert variant_path is not None

    if variant_locks is not None:
        variant_lock = variant_locks.get(variant)
        if variant_lock is None:
            raise ValueError(f"No lock found for build variant: {variant}")
        validate_kernel(repo_path=repo_path, variant=variant, hash=variant_lock.hash)

    module_init_path = variant_path / "__init__.py"
    if not os.path.exists(module_init_path):
        # Compatibility with older kernels.
        module_init_path = variant_path / package_name / "__init__.py"

    if not os.path.exists(module_init_path):
        raise FileNotFoundError(f"No kernel module found at: `{variant_path}`")

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


def get_kernel(
    repo_id: str,
    revision: Optional[str] = None,
    version: Optional[str] = None,
    user_agent: Optional[Union[str, dict]] = None,
) -> ModuleType:
    """
    Load a kernel from the kernel hub.

    This function downloads a kernel to the local Hugging Face Hub cache directory (if it was not downloaded before)
    and then loads the kernel.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`str`, *optional*):
            The kernel version to download. This can be a Python version specifier, such as `">=1.0.0,<2.0.0"`.
            Cannot be used together with `revision`.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.

    Returns:
        `ModuleType`: The imported kernel module.

    Example:
        ```python
        import torch
        from kernels import get_kernel

        activation = get_kernel("kernels-community/activation")
        x = torch.randn(10, 20, device="cuda")
        out = torch.empty_like(x)
        result = activation.silu_and_mul(out, x)
        ```
    """
    revision = select_revision_or_version(repo_id, revision, version)
    package_name, variant_path = install_kernel(
        repo_id, revision=revision, user_agent=user_agent
    )
    return _import_from_path(package_name, variant_path)


def get_local_kernel(repo_path: Path, package_name: str) -> ModuleType:
    """
    Import a kernel from a local kernel repository path.

    Args:
        repo_path (`Path`):
            The local path to the kernel repository.
        package_name (`str`):
            The name of the package to import from the repository.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    # Presume we were given the top level path of the kernel repository.
    for base_path in [repo_path, repo_path / "build"]:
        for v in build_variants():
            variant_path = base_path / v
            if variant_path.exists():
                return _import_from_path(package_name, variant_path)

    # If we didn't find the package in the repo we may have a explicit
    # package path.
    variant_path = repo_path
    if variant_path.exists():
        return _import_from_path(package_name, variant_path)

    raise FileNotFoundError(f"Could not find package '{package_name}' in {repo_path}")


def has_kernel(
    repo_id: str, revision: Optional[str] = None, version: Optional[str] = None
) -> bool:
    """
    Check whether a kernel build exists for the current environment (Torch version and compute framework).

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`str`, *optional*):
            The kernel version to download. This can be a Python version specifier, such as `">=1.0.0,<2.0.0"`.
            Cannot be used together with `revision`.

    Returns:
        `bool`: `True` if a kernel is available for the current environment.
    """
    revision = select_revision_or_version(repo_id, revision, version)

    package_name = package_name_from_repo_id(repo_id)
    variant = build_variant()

    for variant in build_variants():
        for init_file in ["__init__.py", f"{package_name}/__init__.py"]:
            if file_exists(
                repo_id,
                revision=revision,
                filename=f"build/{variant}/{init_file}",
            ):
                return True

    return False


def load_kernel(repo_id: str, *, lockfile: Optional[Path] = None) -> ModuleType:
    """
    Get a pre-downloaded, locked kernel.

    If `lockfile` is not specified, the lockfile will be loaded from the caller's package metadata.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        lockfile (`Path`, *optional*):
            Path to the lockfile. If not provided, the lockfile will be loaded from the caller's package metadata.

    Returns:
        `ModuleType`: The imported kernel module.
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

    allow_patterns = [f"build/{variant}/*" for variant in build_variants()]
    repo_path = Path(
        snapshot_download(
            repo_id,
            allow_patterns=allow_patterns,
            cache_dir=CACHE_DIR,
            revision=locked_sha,
            local_files_only=True,
        )
    )

    try:
        package_name, variant_path = _find_kernel_in_repo_path(
            repo_path, package_name, variant_locks=None
        )
        return _import_from_path(package_name, variant_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Locked kernel `{repo_id}` does not have applicable variant or was not downloaded with `kernels download <project>`"
        )


def get_locked_kernel(repo_id: str, local_files_only: bool = False) -> ModuleType:
    """
    Get a kernel using a lock file.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only use local files and not download from the Hub.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    locked_sha = _get_caller_locked_kernel(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    package_name, variant_path = install_kernel(
        repo_id, locked_sha, local_files_only=local_files_only
    )

    return _import_from_path(package_name, variant_path)


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


def _get_user_agent(
    user_agent: Optional[Union[dict, str]] = None,
) -> Union[None, dict, str]:
    import torch

    from . import __version__

    if os.getenv("DISABLE_TELEMETRY", "false").upper() in ENV_VARS_TRUE_VALUES:
        return None

    glibc = glibc_version()
    python = ".".join(platform.python_version_tuple()[:2])

    if user_agent is None:
        user_agent = {}

    if isinstance(user_agent, dict):
        user_agent.update(
            {
                "kernels": __version__,
                "python": python,
                "torch": torch.__version__,
                "build_variant": build_variant(),
                "file_type": "kernel",
            }
        )
        if glibc is not None:
            user_agent["glibc"] = glibc

    elif isinstance(user_agent, str):
        user_agent += f"; kernels/{__version__}; python/{python}; torch/{torch.__version__}; build_variant/{build_variant()}; file_type/kernel"
        if glibc is not None:
            user_agent += f"; glibc/{glibc}"

    return user_agent
