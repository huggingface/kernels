import logging
from pathlib import Path

from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import GitRefInfo

from kernels._rust import KernelVersion, Oid

logger = logging.getLogger(__name__)


def _cached_refs_dir(repo_id: str) -> Path:
    """The cache directory that holds the refs of a kernel repository."""
    # Lazy import so that we can mock it in tests.
    from kernels.hf_hub import CACHE_DIR

    cache_dir = CACHE_DIR or constants.HF_HUB_CACHE
    return Path(cache_dir) / repo_folder_name(repo_id=repo_id, repo_type="kernel") / "refs"


def _get_available_versions(repo_id: str, *, local_files_only: bool) -> dict[int, GitRefInfo]:
    """Get kernel versions that are available in the repository."""
    from kernels.hf_hub import _get_hf_api

    if local_files_only:
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
    versions: dict[int, GitRefInfo] = {}

    refs_dir = _cached_refs_dir(repo_id)
    if not refs_dir.is_dir():
        return versions

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


def resolve_version_spec_as_ref(repo_id: str, version_spec: int, local_files_only: bool) -> GitRefInfo:
    """
    Get the ref for a kernel with the given version.
    """
    versions = _get_available_versions(repo_id, local_files_only=local_files_only)

    ref = versions.get(version_spec, None)
    if ref is None:
        if local_files_only and not versions:
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


def revision_or_version(*, revision: str | None, version: int | None) -> KernelVersion:
    if revision is not None and version is not None:
        raise ValueError("Only one of `revision` or `version` must be specified.")

    if revision is not None:
        return KernelVersion.Revision(revision)
    elif version is not None:
        return KernelVersion.Version(version)

    raise ValueError(
        "A kernel version or revision must be specified. "
        "Use `version=<major>` for a stable kernel API version or `revision=<branch/tag/commit>` "
        "for an explicit Hub revision. See: https://huggingface.co/docs/kernels/migration"
    )


def _resolve_ref_from_cache(repo_id: str, ref: str) -> str | None:
    """Resolve a ref to a commit using the local Hugging Face cache."""
    refs_dir = _cached_refs_dir(repo_id)
    ref_path = refs_dir / ref

    # A ref is used as a path segment here, so make sure that a ref like
    # `../../elsewhere` cannot read outside the refs directory.
    try:
        ref_path.resolve().relative_to(refs_dir.resolve())
    except (OSError, ValueError):
        return None

    try:
        return ref_path.read_text().strip()
    except OSError:
        return None


def _resolve_ref(repo_id: str, ref: str, *, local_files_only: bool) -> Oid:
    """Resolve a branch, tag, or commit to the commit it points at.

    If the ref is already a full Git SHA, it is returned as-is and does not
    require a Hub request.
    """
    try:
        return Oid.from_str(ref)
    except ValueError:
        pass

    if local_files_only:
        commit = _resolve_ref_from_cache(repo_id, ref)
        if commit is None:
            raise ValueError(
                f"Cannot resolve revision '{ref}' of '{repo_id}' to a commit: the ref is not in "
                "the local cache and Hugging Face Hub is in offline mode. Download the kernel "
                "while online first, or pass an explicit `revision=<commit>`."
            )
    else:
        from kernels.hf_hub import _get_hf_api

        commit = _get_hf_api().repo_info(repo_id=repo_id, repo_type="kernel", revision=ref).sha
        if commit is None:
            raise ValueError(f"Cannot resolve revision '{ref}' of '{repo_id}' to a commit.")

    return Oid.from_str(commit)


def resolve_kernel_version(repo_id: str, version: KernelVersion, *, local_files_only: bool) -> Oid:
    """Resolve a kernel version to the commit it refers to.

    A `KernelVersion` can either be a version number or a revision (branch,
    tag, or commit). This function resolves the version or revision into a
    full Git commit SHA.
    """
    if isinstance(version, KernelVersion.Version):
        ref = resolve_version_spec_as_ref(repo_id, version.version, local_files_only=local_files_only)
        return Oid.from_str(ref.target_commit)
    elif isinstance(version, KernelVersion.Revision):
        return _resolve_ref(repo_id, version.revision, local_files_only=local_files_only)
    else:
        raise ValueError(f"Invalid version type: {version}")


def resolve_revision_or_version(
    repo_id: str,
    *,
    revision: str | None,
    version: int | None,
    local_files_only: bool,
) -> Oid:
    """Resolve a `revision` or `version` to a commit.

    The caller must provide either `revision` or `version`, but not both. The
    revision can be a commit, tag, or branch. The full Git SHA of the revision
    or version is returned."""
    return resolve_kernel_version(
        repo_id,
        revision_or_version(revision=revision, version=version),
        local_files_only=local_files_only,
    )
