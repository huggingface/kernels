import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import Distribution
from pathlib import Path
from types import ModuleType

from huggingface_hub import HfApi, constants
from kernels_data import Metadata

from kernels._system import glibc_version
from kernels._versions import select_revision_or_version
from kernels.backends import _backend, _select_backend
from kernels.compat import has_torch, has_tvm_ffi
from kernels.deps import validate_dependencies
from kernels.lockfile import KernelLock, VariantLock
from kernels.status import resolve_status
from kernels.variants import (
    Variant,
    get_variants,
    get_variants_local,
    resolve_variant,
)

KNOWN_BACKENDS = {"cpu", "cuda", "metal", "neuron", "rocm", "xpu", "npu"}

_ALWAYS_TRUSTED_ORGS = {"kernels-community", "kernels-staging", "kernels-test", "sglang"}


def _check_trust_remote_code(repo_id: str, trust_remote_code: bool | list[str]) -> None:
    """Check whether a kernel repository is trusted.

    When ``trust_remote_code`` is ``False`` (the default), only repositories
    whose publisher is marked as trusted on the Hub are allowed.  Repositories
    from untrusted publishers will raise a ``ValueError``.

    When ``trust_remote_code`` is ``True``, all repositories are allowed.

    When ``trust_remote_code`` is a list of strings, it is treated as a list
    of signing identities to verify against.  Signing verification is not yet
    implemented, so passing a list currently emits a warning and falls back
    to the default trust check (i.e. only trusted publishers are allowed).
    """
    if trust_remote_code is True:
        return

    if isinstance(trust_remote_code, list):
        import warnings

        warnings.warn(
            "Signing identity verification is not yet implemented. "
            "The provided signing identities will be ignored and the "
            "kernel will be treated as untrusted. Use trust_remote_code=True "
            "to bypass trust checks.",
            stacklevel=3,
        )

    org = repo_id.split("/", 1)[0]
    if org in _ALWAYS_TRUSTED_ORGS:
        return

    raise ValueError(
        f"Kernel repository '{repo_id}' is not from a trusted publisher. "
        f"Set trust_remote_code=True to allow loading kernels from untrusted sources."
    )

    # TODO: revisit and update logic when we can check trusted publishers at the
    # user/organization level
    #
    # api = _get_hf_api()
    # try:
    #     info = api.repo_info(repo_id, repo_type="kernel")
    # except Exception as e:
    #     raise ValueError(
    #         f"Could not verify publisher trust status for kernel repository '{repo_id}'. "
    #         "Set trust_remote_code=True to allow loading kernels from untrusted sources."
    #     ) from e

    # if not getattr(info, "trustedPublisher", False):
    #     raise ValueError(
    #         f"Kernel repository '{repo_id}' is not from a trusted publisher. "
    #         f"Set trust_remote_code=True to allow loading kernels from untrusted sources."
    #     )


@dataclass(frozen=True)
class RepoInfo:
    """
    This dataclass stores the origin of the kernel.

    The following fields are available:

    - `repo_id` (`str`): the Hub repository containing the kernel.
    - `revision` (`str`): the specific revision of the kernel.
    """

    repo_id: str
    revision: str


@dataclass(frozen=True)
class LoadedKernel:
    """
    This dataclass provides information about a loaded kernel:

    - `metadata` (`Metadata`): kernel metadata.
    - `module` (`ModuleType`): the imported kernel module.
    - `repo_info` (`kernels.utils.RepoInfo | None`): populated only for
      kernels loaded via `get_kernel`. Loaders that work from a local path
      (`get_local_kernel`) or a lockfile (`get_locked_kernel`, `load_kernel`)
      leave this as `None`.

    The metadata includes the following properties that describe a kernel:

    - `id` (`str`): kernel identifier that is unique to the kernel version + backend.
    - `name` (`str`): the name of the kernel.
    - `version` (`int`): the version of the kernel.
    - `license` (`str`): the license of the kernel.
    - `upstream` (`str | None`): the upstream repository of the kernel.
    - `python_depends` (`list[str]`): required Python dependencies.
    - `backend`: information about the kernel's backend.
    """

    metadata: Metadata
    module: ModuleType
    repo_info: RepoInfo | None


_loaded_kernels: dict[Path, LoadedKernel] = {}


def get_loaded_kernels() -> list[LoadedKernel]:
    """
    Return a snapshot of every kernel that has been loaded into the current process.

    The returned list is a new list; mutating it does not affect the registry.

    Returns:
        `list[LoadedKernel]`: One [`LoadedKernel`] per distinct kernel variant path
        loaded in this process.

    Example:
        ```python
        from kernels import get_kernel, get_loaded_kernels

        get_kernel("kernels-community/activation", version=1)
        for loaded in get_loaded_kernels():
            print(loaded.metadata.name, loaded.repo_info)
        ```
    """
    return list(_loaded_kernels.values())


def _get_cache_dir() -> str | None:
    """Returns the kernels cache directory."""
    return os.environ.get("KERNELS_CACHE", None)


def _get_local_kernel_overrides() -> dict[str, Path]:
    """Returns list local overrides for kernels."""
    local_kerels = os.environ.get("LOCAL_KERNELS", None)
    if local_kerels is None:
        return dict()
    return _parse_local_kernel_overrides(local_kerels)


@functools.lru_cache(maxsize=1)
def _parse_local_kernel_overrides(local_kernels: str) -> dict[str, Path]:
    """Parse the LOCAL_KERNELS environment variable into a dictionary."""
    overrides = {}
    for entry in local_kernels.split(":"):
        if "=" not in entry:
            raise ValueError(
                f"Invalid LOCAL_KERNELS entry: {entry}. Expected format: repo_id_1=path_1:repo_id_2=path_2"
            )
        repo_id, path = entry.split("=", 1)
        overrides[repo_id] = Path(path)

    return overrides


CACHE_DIR: str | None = _get_cache_dir()


def _import_from_path(variant_path: Path, repo_info: RepoInfo | None = None) -> ModuleType:
    if (loaded_kernel := _loaded_kernels.get(variant_path)) is not None:
        return loaded_kernel.module

    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    module_name = metadata.name.python_name
    validate_dependencies(module_name, metadata.python_depends, _backend())

    file_path = variant_path / "__init__.py"
    if not file_path.exists():
        file_path = variant_path / module_name / "__init__.py"
    if not file_path.exists():
        raise FileNotFoundError(f"No kernel module found at: `{variant_path}`")

    spec = importlib.util.spec_from_file_location(metadata.id, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Cannot load module {module_name} from spec")
    sys.modules[metadata.id] = module
    spec.loader.exec_module(module)  # type: ignore

    _loaded_kernels[variant_path] = LoadedKernel(
        metadata=metadata,
        module=module,
        repo_info=repo_info,
    )
    return module


def install_kernel(
    repo_id: str,
    *,
    revision: str,
    local_files_only: bool = False,
    backend: str | None = None,
    variant_locks: dict[str, VariantLock] | None = None,
    user_agent: str | dict | None = None,
) -> Path:
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
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.
        variant_locks (`dict[str, VariantLock]`, *optional*):
            Optional dictionary of variant locks for validation.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.

    Returns:
        `Path`: The path to the variant directory.
    """
    api = _get_hf_api(user_agent=user_agent)

    if not local_files_only:
        repo_id, revision = resolve_status(api, repo_id, revision)

    variants = get_variants(api, repo_id=repo_id, revision=revision)
    variant = resolve_variant(variants, backend)

    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}). Available variants: {', '.join([variant.variant_str for variant in variants])}"
        )

    allow_patterns = [f"build/{variant.variant_str}/*"]

    repo_path = Path(
        str(
            api.snapshot_download(
                repo_id,
                repo_type="kernel",
                allow_patterns=allow_patterns,
                cache_dir=CACHE_DIR,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    )

    try:
        return _find_kernel_in_repo_path(
            repo_path,
            variant=variant,
            variant_locks=variant_locks,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot install kernel from repo {repo_id} (revision: {revision})")


def _find_kernel_in_repo_path(
    repo_path: Path,
    *,
    variant: Variant,
    variant_locks: dict[str, VariantLock] | None = None,
) -> Path:
    variant_str = variant.variant_str
    variant_path = repo_path / "build" / variant_str
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant path does not exist: `{variant_path}`")

    if variant_locks is not None:
        variant_lock = variant_locks.get(variant_str)
        if variant_lock is None:
            raise ValueError(f"No lock found for build variant: {variant}")
        validate_kernel(repo_path=repo_path, variant=variant_str, hash=variant_lock.hash)

    return variant_path


def install_kernel_all_variants(
    repo_id: str,
    *,
    revision: str,
    local_files_only: bool = False,
    variant_locks: dict[str, VariantLock] | None = None,
) -> Path:
    api = _get_hf_api()

    repo_path = Path(
        str(
            api.snapshot_download(
                repo_id,
                repo_type="kernel",
                allow_patterns="build/*",
                cache_dir=CACHE_DIR,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    )

    if variant_locks is not None:
        for entry in (repo_path / "build").iterdir():
            variant = entry.parts[-1]

            variant_lock = variant_locks.get(variant)
            if variant_lock is None:
                raise ValueError(f"No lock found for build variant: {variant}")

            validate_kernel(repo_path=repo_path, variant=variant, hash=variant_lock.hash)

    return repo_path / "build"


def get_kernel(
    repo_id: str,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    user_agent: str | dict | None = None,
    trust_remote_code: bool | list[str] = False,
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
        version (`int`, *optional*):
            The kernel version to download. Cannot be used together with `revision`.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.
        trust_remote_code (`bool | list[str]`, *optional*, defaults to `False`):
            Whether to allow loading kernels from untrusted organisations. When ``False``,
            only kernels from trusted organisations are allowed. When ``True``, all
            repositories are allowed. A list of strings will be used to verify signing
            identities in a future release; for now it emits a warning and falls
            back to the default trust check.

    Returns:
        `ModuleType`: The imported kernel module.

    Example:
        ```python
        import torch
        from kernels import get_kernel

        activation = get_kernel("kernels-community/relu", version=1)
        x = torch.randn(10, 20, device="cuda")
        out = torch.empty_like(x)
        result = activation.relu(out, x)
        ```
    """
    override = _get_local_kernel_overrides().get(repo_id, None)
    if override is not None:
        return get_local_kernel(override)

    _check_trust_remote_code(repo_id, trust_remote_code)

    revision = select_revision_or_version(repo_id, revision=revision, version=version)
    repo_info = RepoInfo(
        repo_id=repo_id,
        revision=revision,
    )
    variant_path = install_kernel(
        repo_id,
        backend=backend,
        revision=revision,
        user_agent=user_agent,
    )
    return _import_from_path(variant_path, repo_info=repo_info)


def get_local_kernel(
    repo_path: Path,
    backend: str | None = None,
) -> ModuleType:
    """
    Import a kernel from a local kernel repository path.

    Args:
        repo_path (`Path`):
            The local path to the kernel repository.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    for base_path in [repo_path, repo_path / "build"]:
        variants = get_variants_local(base_path)
        variant = resolve_variant(variants, backend)

        if variant is not None:
            return _import_from_path(base_path / variant.variant_str)

    # If we didn't find the package in the repo we may have a explicit
    # package path.
    variant_path = repo_path
    if variant_path.exists():
        return _import_from_path(variant_path)

    raise FileNotFoundError(f"Could not find kernel in {repo_path}")


def has_kernel(
    repo_id: str,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
) -> bool:
    """
    Check whether a kernel build exists for the current environment (Torch version and compute framework).

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`int`, *optional*):
            The kernel version to download. Cannot be used together with `revision`.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.

    Returns:
        `bool`: `True` if a kernel is available for the current environment.
    """
    revision = select_revision_or_version(repo_id, revision=revision, version=version)

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    variant = resolve_variant(variants, backend)

    if variant is None:
        return False

    return api.file_exists(
        repo_id,
        repo_type="kernel",
        revision=revision,
        filename=f"build/{variant.variant_str}/metadata.json",
    )


def load_kernel(
    repo_id: str,
    *,
    lockfile: Path | None,
    backend: str | None = None,
) -> ModuleType:
    """
    Get a pre-downloaded, locked kernel.

    If `lockfile` is not specified, the lockfile will be loaded from the caller's package metadata.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        lockfile (`Path`, *optional*):
            Path to the lockfile. If not provided, the lockfile will be loaded from the caller's package metadata.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.

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

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=locked_sha)
    variant = resolve_variant(variants, backend)

    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {locked_sha}). Available variants: {', '.join([variant.variant_str for variant in variants])}"
        )

    allow_patterns = [f"build/{variant.variant_str}/*"]
    repo_path = Path(
        str(
            api.snapshot_download(
                repo_id,
                repo_type="kernel",
                allow_patterns=allow_patterns,
                cache_dir=CACHE_DIR,
                revision=locked_sha,
                local_files_only=True,
            )
        )
    )

    try:
        variant_path = _find_kernel_in_repo_path(
            repo_path,
            variant=variant,
            variant_locks=None,
        )
        return _import_from_path(variant_path)
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

    variant_path = install_kernel(repo_id, revision=locked_sha, local_files_only=local_files_only)

    return _import_from_path(variant_path)


def _get_caller_locked_kernel(repo_id: str) -> str | None:
    for dist in _get_caller_distributions():
        lock_json = dist.read_text("kernels.lock")
        if lock_json is None:
            continue
        locked_sha = _get_locked_kernel(repo_id, lock_json)
        if locked_sha is not None:
            return locked_sha
    return None


def _get_locked_kernel(repo_id: str, lock_json: str) -> str | None:
    for kernel_lock_json in json.loads(lock_json):
        kernel_lock = KernelLock.from_json(kernel_lock_json)
        if kernel_lock.repo_id == repo_id:
            return kernel_lock.sha
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


def validate_kernel(*, repo_path: Path, variant: str, hash: str):
    """Validate the given build variant of a kernel against a hasht."""
    variant_path = repo_path / "build" / variant

    # Get the file paths. The first element is a byte-encoded relative path
    # used for sorting. The second element is the absolute path.
    files: list[tuple[bytes, Path]] = []
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


def _platform() -> str:
    cpu = platform.machine()
    os = platform.system().lower()

    if os == "darwin":
        cpu = "aarch64" if cpu == "arm64" else cpu
    elif os == "windows":
        cpu = "x86_64" if cpu == "AMD64" else cpu

    return f"{cpu}-{os}"


def _get_hf_api(user_agent: str | dict | None = None) -> HfApi:
    """Returns an instance of HfApi with proper settings."""

    from . import __version__

    user_agent_str = ""
    if not constants.HF_HUB_DISABLE_TELEMETRY:
        # User-defined info
        if isinstance(user_agent, dict):
            user_agent_str = "; ".join(f"{k}/{v}" for k, v in user_agent.items())
        if isinstance(user_agent, str):
            user_agent_str = user_agent

        # System info
        python = ".".join(platform.python_version_tuple()[:2])
        backend = _select_backend(None).variant_str
        user_agent_str += (
            f"; kernels/{__version__}; python/{python}; backend/{backend}; platform/{_platform()}; file_type/kernel"
        )

        if has_torch:
            import torch

            user_agent_str += f"; torch/{torch.__version__}"
        if has_tvm_ffi:
            import tvm_ffi

            user_agent_str += f"; tvm-ffi/{tvm_ffi.__version__}"

        # Add glibc version if available
        glibc = glibc_version()
        if glibc is not None:
            user_agent_str += f"; glibc/{glibc}"

    return HfApi(library_name="kernels", library_version=__version__, user_agent=user_agent_str)
