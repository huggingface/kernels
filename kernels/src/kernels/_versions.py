import logging
import os
from pathlib import Path

from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import GitRefInfo

logger = logging.getLogger(__name__)


def _get_available_versions(repo_id: str) -> dict[int, GitRefInfo]:
    """Get kernel versions that are available in the repository."""
    from kernels.utils import _get_hf_api

    if constants.HF_HUB_OFFLINE:
        return _get_available_versions_from_cache(repo_id)

    refs = _get_hf_api().list_repo_refs(repo_id=repo_id, repo_type="kernel")

    versions = {}
    for branch in refs.branches:
        if not branch.name.startswith("v"):
            continue
        try:
            versions[int(branch.name[1:])] = branch
        except ValueError:
            continue

    return versions


def _get_available_versions_from_cache(repo_id: str) -> dict[int, GitRefInfo]:
    """Get kernel versions from the local Hugging Face cache."""
    cache_dir = os.environ.get("KERNELS_CACHE") or constants.HF_HUB_CACHE

    versions: dict[int, GitRefInfo] = {}
    # Tolerate both layouts: the "kernel" repo type used by newer
    # huggingface_hub, and the legacy "model" prefix that older caches use.
    for repo_type in ("kernel", "model"):
        refs_dir = Path(cache_dir) / repo_folder_name(repo_id=repo_id, repo_type=repo_type) / "refs"
        if not refs_dir.is_dir():
            continue
        for ref_path in refs_dir.iterdir():
            if not ref_path.is_file():
                continue
            ref_name = ref_path.name
            if not ref_name.startswith("v"):
                continue
            try:
                version = int(ref_name[1:])
            except ValueError:
                continue
            try:
                commit = ref_path.read_text().strip()
            except OSError:
                continue
            versions[version] = GitRefInfo(name=ref_name, ref=ref_name, target_commit=commit)

    return versions


def resolve_version_spec_as_ref(repo_id: str, version_spec: int) -> GitRefInfo:
    """
    Get the ref for a kernel with the given version.
    """
    versions = _get_available_versions(repo_id)

    ref = versions.get(version_spec, None)
    if ref is None:
        if constants.HF_HUB_OFFLINE and not versions:
            raise ValueError(
                f"Version {version_spec} of '{repo_id}' is not available in the local cache "
                "and Hugging Face Hub is in offline mode. Download the kernel "
                "while online first, or pass an explicit `revision=<commit>`."
            )
        raise ValueError(
            f"Version {version_spec} not found, available versions: {', '.join(str(v) for v in sorted(versions.keys()))}"
        )

    latest_version = max(versions.keys())
    if version_spec < latest_version:
        logger.warning(
            "You are using version %d of '%s', but version %d is available.",
            version_spec,
            repo_id,
            latest_version,
        )

    return ref


def select_revision_or_version(
    repo_id: str,
    *,
    revision: str | None,
    version: int | None,
) -> str:
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")
    elif revision is not None:
        return revision
    elif version is not None:
        return resolve_version_spec_as_ref(repo_id, version).target_commit
    else:
        raise ValueError(
            "A kernel version or revision must be specified. "
            "Use `version=<major>` for a stable kernel API version or `revision=<branch/tag/commit>` "
            "for an explicit Hub revision. See: https://huggingface.co/docs/kernels/migration"
        )
