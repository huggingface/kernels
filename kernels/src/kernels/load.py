import functools
import os
from pathlib import Path
from types import ModuleType

from huggingface_hub import constants

from kernels._versions import select_revision_or_version
from kernels.deps import validate_variant_dependencies
from kernels.hf_hub import RepoInfo, _check_trust_remote_code, _get_hf_api
from kernels.importer import _import_from_path
from kernels.install import install_kernel
from kernels.locking import (
    get_caller_locked_kernel_revision,
    get_locked_kernel_revision,
)
from kernels.variants import (
    get_variants,
    get_variants_local,
    resolve_variant,
)


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
        revision (`str`, *optional*):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`int`, *optional*):
            The kernel version to download. Cannot be used together with `revision`.
            Either `version` or `revision` must be specified.
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

    _check_trust_remote_code(
        repo_id=repo_id,
        local_files_only=constants.HF_HUB_OFFLINE,
        trust_remote_code=trust_remote_code,
    )

    revision = select_revision_or_version(
        repo_id,
        revision=revision,
        version=version,
        local_files_only=constants.HF_HUB_OFFLINE,
    )
    repo_info = RepoInfo(
        repo_id=repo_id,
        revision=revision,
    )
    variant_path = install_kernel(
        repo_id,
        backend=backend,
        revision=revision,
        user_agent=user_agent,
        validate_dependencies=True,
        local_files_only=constants.HF_HUB_OFFLINE,
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
        variant, _ = resolve_variant(variants, backend)

        if variant is not None:
            variant_path = base_path / variant.variant_str
            validate_variant_dependencies(variant_path)
            return _import_from_path(variant_path)

    # If we didn't find the package in the repo we may have a explicit
    # package path.
    variant_path = repo_path
    if variant_path.exists():
        validate_variant_dependencies(variant_path)
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
        revision (`str`, *optional*):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`int`, *optional*):
            The kernel version to download. Cannot be used together with `revision`.
            Either `version` or `revision` must be specified.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.

    Returns:
        `bool`: `True` if a kernel is available for the current environment.
    """
    revision = select_revision_or_version(
        repo_id,
        revision=revision,
        version=version,
        local_files_only=constants.HF_HUB_OFFLINE,
    )

    api = _get_hf_api()
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    variant, _ = resolve_variant(variants, backend)

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
        locked_sha = get_caller_locked_kernel_revision(repo_id)
    else:
        with open(lockfile, "r") as f:
            locked_sha = get_locked_kernel_revision(repo_id, f.read())

    if locked_sha is None:
        raise ValueError(
            f"Kernel `{repo_id}` is not locked. Please lock it with `kernels lock <project>` and then reinstall the project."
        )

    try:
        variant_path = install_kernel(repo_id, revision=locked_sha, backend=backend, local_files_only=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Locked kernel `{repo_id}` was not downloaded or does not have an "
            "applicable variant. Make sure it's downloaded locally via "
            "`kernels download <project>`."
        ) from e
    return _import_from_path(variant_path)


def get_locked_kernel(repo_id: str) -> ModuleType:
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
    locked_sha = get_caller_locked_kernel_revision(repo_id)

    if locked_sha is None:
        raise ValueError(f"Kernel `{repo_id}` is not locked")

    variant_path = install_kernel(
        repo_id,
        revision=locked_sha,
        local_files_only=constants.HF_HUB_OFFLINE,
        validate_dependencies=True,
    )

    return _import_from_path(variant_path)
