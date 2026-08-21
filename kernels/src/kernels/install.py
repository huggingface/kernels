from pathlib import Path

from kernels_data import KernelDependency

from kernels._versions import revision_or_version
from kernels.deps import resolve_kernel_tree
from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.locking import extract_dependency_locks
from kernels.resolver import _BYTECODE_IGNORE_PATTERNS, HubCacheResolver, HubResolver


def install_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    local_files_only: bool = False,
    user_agent: str | dict | None = None,
    trust_remote_code: bool | list[str] = False,
) -> Path:
    """
    Download a kernel for the current environment to the cache.

    The output path is validated against the hashes in `variant_locks` when provided.

    Args:
        repo_id (`str`):
            The Hub repository containing the kernel.
        revision (`str`):
            The specific revision (branch, tag, or commit) to download.
        backend (`str`, *optional*):
            The backend to load the kernel for. Can only be `cpu` or the backend that Torch is compiled for.
            The backend will be detected automatically if not provided.
        local_files_only (`bool`, *optional*, defaults to `False`):
            Whether to only use local files and not download from the Hub.
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.
        trust_remote_code (`bool | list[str]`, *optional*, defaults to `False`):
            Whether to allow loading kernels from untrusted organisations. When ``False``,
            only kernels from trusted organisations are allowed. When ``True``, all
            repositories are allowed. A list of strings will be used to verify signing
            identities in a future release; for now it emits a warning and falls
            back to the default trust check.

    Returns:
        `Path`: The path to the variant directory.
    """
    api = _get_hf_api(user_agent=user_agent)

    kernel_version = revision_or_version(revision=revision, version=version)

    resolver = (
        HubCacheResolver(trust_remote_code=trust_remote_code)
        if local_files_only
        else HubResolver(trust_remote_code=trust_remote_code)
    )

    tree = resolve_kernel_tree(
        api=api,
        backend=backend,
        kernel=KernelDependency(repo_id=repo_id, version=kernel_version),
        resolver=resolver,
    )

    return tree.install(api=api).location.variant_path


def install_kernel_all_variants(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
) -> Path:
    kernel_dep = KernelDependency(repo_id=repo_id, version=revision_or_version(revision=revision, version=version))

    # Use locking code path to recursively get dependencies.
    api = _get_hf_api()
    locks = extract_dependency_locks(
        [kernel_dep],
        api=api,
        backend=None,
    )

    paths = {}
    for dep, lock in locks.items():
        paths[dep] = Path(
            str(
                api.snapshot_download(
                    dep.repo_id,
                    repo_type="kernel",
                    allow_patterns="build/*",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=lock.commit,
                )
            )
        )

    return paths[kernel_dep] / "build"
