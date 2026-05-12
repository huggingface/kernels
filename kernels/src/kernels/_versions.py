import logging

from huggingface_hub.hf_api import GitRefInfo

logger = logging.getLogger(__name__)


def _get_available_versions(repo_id: str) -> dict[int, GitRefInfo]:
    """Get kernel versions that are available in the repository."""
    from kernels.utils import _get_hf_api

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


def resolve_version_spec_as_ref(repo_id: str, version_spec: int) -> GitRefInfo:
    """
    Get the ref for a kernel with the given version.
    """
    versions = _get_available_versions(repo_id)

    ref = versions.get(version_spec, None)
    if ref is None:
        raise ValueError(
            f"Version {version_spec} not found, available versions: {', '.join(sorted(str(v) for v in versions.keys()))}"
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

    if revision is not None:
        return revision

    if version is not None:
        return resolve_version_spec_as_ref(repo_id, version).target_commit

    raise ValueError(
        "A kernel version or revision must be specified. "
        "Use `version=<major>` for a stable kernel API version or `revision=<branch/tag/commit>` "
        "for an explicit Hub revision. See: https://huggingface.co/docs/kernels/migration"
    )
