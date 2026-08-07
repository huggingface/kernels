from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import LocalEntryNotFoundError

from kernels.hf_hub import CACHE_DIR, _get_hf_api
from kernels.python_deps import validate_variant_dependencies
from kernels.status import resolve_status
from kernels.variants import (
    Variant,
    get_variants,
    get_variants_local,
    resolve_variant,
    variants_trace_str,
)

# Exclude pattern for bytecode. These are not included in kernel builds,
# but builds not done using kernel-builder might accidentally include
# bytecode. So these patterns are used to ensure that they are never
# downloaded.
_BYTECODE_IGNORE_PATTERNS = ["*.pyc", "**/__pycache__/**"]


def install_kernel(
    repo_id: str,
    *,
    revision: str,
    local_files_only: bool = False,
    backend: str | None = None,
    user_agent: str | dict | None = None,
    validate_dependencies: bool = False,
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
        user_agent (`Union[str, dict]`, *optional*):
            The `user_agent` info to pass to `snapshot_download()` for internal telemetry.
        validate_dependencies (`bool`, defaults to False):
            When set to True, performs dependency validation. Useful for kernels that have
            extra Python dependencies.

    Returns:
        `Path`: The path to the variant directory.
    """
    api = _get_hf_api(user_agent=user_agent)
    if local_files_only:
        # Same local-cache resolution path used by `load_kernel`, which is
        # always offline. Sharing the helper avoids the network dependency
        # that `get_variants` would otherwise introduce.
        variant_path = _resolve_local_variant_path(
            api,
            repo_id,
            revision=revision,
            backend=backend,
        )
        # For locally downloaded kernels, we run the validation after resolving the path
        if validate_dependencies:
            validate_variant_dependencies(variant_path)
        return variant_path

    repo_id, revision = resolve_status(api, repo_id, revision)
    variants = get_variants(api, repo_id=repo_id, revision=revision)
    variant, trace = resolve_variant(variants, backend)

    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}):\n\n{variants_trace_str(trace)}"
        )

    # Validate Python dependencies before downloading the variant.
    if validate_dependencies:
        metadata_path = api.hf_hub_download(
            repo_id,
            repo_type="kernel",
            filename=f"build/{variant.variant_str}/metadata.json",
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=False,
        )
        validate_variant_dependencies(Path(metadata_path).parent)

    allow_patterns = [f"build/{variant.variant_str}/*"]
    ignore_patterns = _BYTECODE_IGNORE_PATTERNS

    repo_path = Path(
        str(
            api.snapshot_download(
                repo_id,
                repo_type="kernel",
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=CACHE_DIR,
                revision=revision,
                local_files_only=False,
            )
        )
    )

    try:
        return _find_kernel_in_repo_path(
            repo_path,
            variant=variant,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot install kernel from repo {repo_id} (revision: {revision})")


def _resolve_local_variant_path(
    api: HfApi,
    repo_id: str,
    *,
    revision: str,
    backend: str | None = None,
) -> Path:
    """Resolve a kernel variant path from the local Hugging Face cache only.

    Used by `load_kernel` (which always operates on a pre-downloaded, locked
    kernel) and by the offline branch of `install_kernel`.
    """
    try:
        local_repo_path = Path(
            str(
                api.snapshot_download(
                    repo_id,
                    repo_type="kernel",
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=revision,
                    local_files_only=True,
                )
            )
        )
    except LocalEntryNotFoundError as e:
        raise FileNotFoundError(
            f"Cannot find a local snapshot for {repo_id} (revision: {revision}). "
            "When Hugging Face Hub is in offline mode the kernel must already "
            "be present in the local cache."
        ) from e

    variants = get_variants_local(local_repo_path / "build")
    variant, status = resolve_variant(variants, backend)
    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}):\n\n{variants_trace_str(status)}"
        )

    return _find_kernel_in_repo_path(
        local_repo_path,
        variant=variant,
    )


def _find_kernel_in_repo_path(
    repo_path: Path,
    *,
    variant: Variant,
) -> Path:
    variant_str = variant.variant_str
    variant_path = repo_path / "build" / variant_str
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant path does not exist: `{variant_path}`")

    return variant_path


def install_kernel_all_variants(
    repo_id: str,
    *,
    revision: str,
) -> Path:
    api = _get_hf_api()

    repo_path = Path(
        str(
            api.snapshot_download(
                repo_id,
                repo_type="kernel",
                allow_patterns="build/*",
                ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                cache_dir=CACHE_DIR,
                revision=revision,
            )
        )
    )

    return repo_path / "build"
