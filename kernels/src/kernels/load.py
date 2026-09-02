import functools
import os
from pathlib import Path
from types import ModuleType

from huggingface_hub import HfApi, constants
from kernels_data import KernelDependency, KernelVersion

from kernels._versions import revision_or_version
from kernels.deps import resolve_kernel_tree
from kernels.hf_hub import _get_hf_api
from kernels.locking import (
    get_caller_locked_kernel_revision,
    get_locked_kernel_revision,
)
from kernels.resolver import (
    HubCacheResolver,
    HubResolver,
    LockedHubCacheResolver,
    LockedHubResolver,
    NoopResolver,
    RepoPathsResolver,
    Resolver,
    SequentialResolver,
)
from kernels.validate import (
    AllValidator,
    ArchValidator,
    MetadataValidator,
    default_metadata_validators,
)


def _get_local_kernel_overrides() -> Resolver:
    """Returns list local overrides for kernels."""
    local_kerels = os.environ.get("LOCAL_KERNELS", None)
    if local_kerels is None:
        return NoopResolver()
    overrides = _parse_local_kernel_overrides(local_kerels)
    return RepoPathsResolver(local_kernels=overrides)


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


def get_kernel_with_resolver(
    *,
    api: HfApi,
    backend: str | None,
    kernel: KernelDependency,
    resolver: Resolver | None,
    validator: MetadataValidator,
) -> ModuleType:
    """
    Load a kernel and its (transitive) dependencies using the given resolver.

    Args:
        api (`HfApi`):
            The Hugging Face Hub API client.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.
        kernel (`KernelDependency`):
            The kernel to load.
        resolver (`Resolver`, *optional*):
            The resolver used to resolve the kernel and its (transitive) dependencies.
        validator (`MetadataValidator`):
            The validator to apply to the resolved kernel dependency tree.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    tree = resolve_kernel_tree(
        api=api,
        backend=backend,
        kernel=kernel,
        resolver=resolver,
    )
    tree.validate_metadata(validator)
    tree_only_local = tree.install(api=api)
    return tree_only_local.load()


def get_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    user_agent: str | dict | None = None,
    trust_remote_code: bool | list[str] = False,
    check_arch: bool = True,
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
        check_arch (`bool`, *optional*, defaults to `True`):
            Whether to check that the kernel build supports the architecture
            (e.g. CUDA compute capability) of the current device. Kernels can
            support more architectures than they declare (e.g. through a
            Triton fallback), `check_arch=False` skips the check for such
            kernels.

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
    api = _get_hf_api(user_agent=user_agent)

    kernel_version = revision_or_version(revision=revision, version=version)

    resolvers = [
        _get_local_kernel_overrides(),
        (
            HubCacheResolver(trust_remote_code=trust_remote_code)
            if constants.HF_HUB_OFFLINE
            else HubResolver(trust_remote_code=trust_remote_code)
        ),
    ]

    validators: list[MetadataValidator] = default_metadata_validators()
    if check_arch:
        validators.append(ArchValidator())

    return get_kernel_with_resolver(
        api=api,
        backend=backend,
        kernel=KernelDependency(repo_id=repo_id, version=kernel_version),
        resolver=SequentialResolver(resolvers=resolvers),
        validator=AllValidator(validators=validators),
    )


def get_local_kernel(
    repo_path: Path,
    *,
    backend: str | None = None,
    trust_remote_code: bool | list[str] = False,
    user_agent: str | dict | None = None,
) -> ModuleType:
    """
    Import a kernel from a local kernel repository path.

    If the kernel has any (transitive) dependencies, they will be downloaded.

    Args:
        repo_path (`Path`):
            The local path to the kernel repository.
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
    """
    api = _get_hf_api(user_agent=user_agent)

    resolvers = [
        RepoPathsResolver(local_kernels={str(repo_path): repo_path}),
        _get_local_kernel_overrides(),
        (
            HubCacheResolver(trust_remote_code=trust_remote_code)
            if constants.HF_HUB_OFFLINE
            else HubResolver(trust_remote_code=trust_remote_code)
        ),
    ]

    return get_kernel_with_resolver(
        api=api,
        backend=backend,
        # We don't have a name for the kernel, so let's just use the path.
        kernel=KernelDependency(repo_id=str(repo_path), version=KernelVersion.Version(0)),
        resolver=SequentialResolver(resolvers),
        validator=AllValidator(validators=default_metadata_validators()),
    )


def has_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    trust_remote_code: bool | list[str] = False,
    check_arch: bool = True,
) -> bool:
    """
    Check whether a kernel build exists for the current environment (framework version and backend).

    If the kernel has any (transitive) dependencies, they will be checked as well.

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
        trust_remote_code (`bool | list[str]`, *optional*, defaults to `False`):
            Whether to allow loading kernels from untrusted organisations. When ``False``,
            only kernels from trusted organisations are allowed. When ``True``, all
            repositories are allowed. A list of strings will be used to verify signing
            identities in a future release; for now it emits a warning and falls
            back to the default trust check.
        check_arch (`bool`, *optional*, defaults to `True`):
            Whether to check that the kernel build supports the architecture
            (e.g. CUDA compute capability) of the current device. Kernels can
            support more architectures than they declare (e.g. through a
            Triton fallback), `check_arch=False` skips the check for such
            kernels.

    Returns:
        `bool`: `True` if a kernel is available for the current environment.
    """
    api = _get_hf_api()

    kernel_version = revision_or_version(revision=revision, version=version)

    resolvers = [
        _get_local_kernel_overrides(),
        (
            HubCacheResolver(trust_remote_code=trust_remote_code)
            if constants.HF_HUB_OFFLINE
            else HubResolver(trust_remote_code=trust_remote_code)
        ),
    ]

    try:
        tree = resolve_kernel_tree(
            api=api,
            backend=backend,
            kernel=KernelDependency(repo_id=repo_id, version=kernel_version),
            resolver=SequentialResolver(resolvers=resolvers),
        )
    except FileNotFoundError:
        return False

    if check_arch:
        try:
            tree.validate_metadata(ArchValidator())
        except RuntimeError:
            return False

    return True


def load_kernel(
    repo_id: str,
    *,
    lockfile: Path | None,
    backend: str | None = None,
) -> ModuleType:
    """
    Get a pre-downloaded, locked kernel.

    This function will never download anything and will fail when a kernel is
    not available locally.

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
        kernel_locks, kernel_dep = get_caller_locked_kernel_revision(repo_id)
    else:
        kernel_locks, kernel_dep = get_locked_kernel_revision(repo_id, lockfile)

    resolver = LockedHubCacheResolver(kernel_locks=kernel_locks, trust_remote_code=False)

    return get_kernel_with_resolver(
        # We need to provide the API, but it is never used, so no need for a
        # user agent.
        api=_get_hf_api(),
        backend=backend,
        kernel=kernel_dep,
        resolver=resolver,
        validator=AllValidator(validators=default_metadata_validators()),
    )


def get_locked_kernel(
    repo_id: str,
    *,
    lockfile: Path | None,
    trust_remote_code: bool | list[str] = False,
    user_agent: str | dict | None = None,
) -> ModuleType:
    """
    Get a kernel using a lock file.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        lockfile (`Path`, *optional*):
            Path to the lockfile. If not provided, the lockfile will be loaded from the caller's package metadata.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.

    Returns:
        `ModuleType`: The imported kernel module.
    """
    if lockfile is None:
        kernel_locks, kernel_dep = get_caller_locked_kernel_revision(repo_id)
    else:
        kernel_locks, kernel_dep = get_locked_kernel_revision(repo_id, lockfile)

    resolver = (
        LockedHubCacheResolver(kernel_locks=kernel_locks, trust_remote_code=trust_remote_code)
        if constants.HF_HUB_OFFLINE
        else LockedHubResolver(kernel_locks=kernel_locks, trust_remote_code=trust_remote_code)
    )

    return get_kernel_with_resolver(
        api=_get_hf_api(user_agent=user_agent),
        backend=None,
        kernel=kernel_dep,
        resolver=resolver,
        validator=AllValidator(validators=default_metadata_validators()),
    )
