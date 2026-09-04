from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.hf_api import HfApi
from kernels_data import KernelDependency, KernelLocks, KernelPaths, Metadata

from kernels._versions import _get_available_versions, resolve_kernel_version
from kernels.hf_hub import CACHE_DIR, _check_trust_remote_code
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


@dataclass(frozen=True)
class LocalKernel:
    """
    A kernel that can be loaded from a local path.

    `origin` is `None` if the kernel was a local kernel before installation,
    otherwise it will point to the remote kernel that was intalled.
    """

    variant_path: Path
    metadata: Metadata
    origin: "RemoteKernel | None" = None

    @property
    def variant_str(self) -> str:
        return self.variant_path.name

    def install(self, *, api: HfApi) -> "LocalKernel":
        # Local kernels are already installed, so we just return self.
        return self


@dataclass(frozen=True)
class RemoteKernel:
    """A kernel that can be loaded from a remote path."""

    repo_id: str
    revision: str
    metadata: Metadata
    variant: Variant

    @property
    def variant_str(self) -> str:
        return self.variant.variant_str

    def install(self, *, api: HfApi) -> LocalKernel:
        allow_patterns = [f"build/{self.variant.variant_str}/*"]

        repo_path = Path(
            str(
                api.snapshot_download(
                    self.repo_id,
                    repo_type="kernel",
                    allow_patterns=allow_patterns,
                    ignore_patterns=_BYTECODE_IGNORE_PATTERNS,
                    cache_dir=CACHE_DIR,
                    revision=self.revision,
                    local_files_only=False,
                )
            )
        )

        return LocalKernel(
            variant_path=repo_path / "build" / self.variant.variant_str,
            metadata=self.metadata,
            origin=self,
        )


def resolve_local_kernel(repo_path: Path, *, backend: str | None) -> LocalKernel:
    variant_path = None

    for base_path in [repo_path, repo_path / "build"]:
        variants = get_variants_local(base_path)
        variant, trace = resolve_variant(variants, backend)
        if variant is not None:
            variant_path = base_path / variant.variant_str
            break

    # The local kernel path might be pointing directly to the variant.
    if variant_path is None:
        metadata_path = repo_path / "metadata.json"
        if metadata_path.exists():
            variant_path = repo_path

    if variant_path is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_path}:\n\n{variants_trace_str(trace)}"
        )

    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    location = LocalKernel(variant_path=variant_path, metadata=metadata)

    return location


def resolve_hub_kernel(
    repo_id: str,
    *,
    api: HfApi,
    backend: str | None,
    revision: str,
) -> RemoteKernel:
    variants = get_variants(
        api,
        repo_id=repo_id,
        revision=revision,
    )
    variant, trace = resolve_variant(variants, backend)
    if variant is None:
        suggestion = _latest_compatible_version_suggestion(
            api=api,
            repo_id=repo_id,
            backend=backend,
        )
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}):\n\n"
            f"{variants_trace_str(trace)}{suggestion}"
        )

    metadata_path = Path(
        api.hf_hub_download(
            repo_id,
            repo_type="kernel",
            filename=f"build/{variant.variant_str}/metadata.json",
            cache_dir=CACHE_DIR,
            revision=revision,
            local_files_only=False,
        )
    )

    metadata = Metadata.read_from_file(metadata_path)
    location = RemoteKernel(repo_id=repo_id, revision=revision, variant=variant, metadata=metadata)

    return location


def _latest_compatible_version_suggestion(
    *,
    api: HfApi,
    repo_id: str,
    backend: str | None,
) -> str:
    """
    This runs only after variant resolution has failed. Version discovery and
    variant inspection involve additional Hub requests, so the lookup is
    best-effort and hence, we never mask the original resolution error.
    """
    try:
        versions = _get_available_versions(repo_id, local_files_only=False)
        if not versions:
            return ""

        latest_version = max(versions)
        ref = versions[latest_version]
        variants = get_variants(api, repo_id=repo_id, revision=ref.ref)
        compatible_variant, _ = resolve_variant(variants, backend)
        if compatible_variant is not None:
            return (
                f"\n\nHowever, version v{latest_version} of '{repo_id}' has a build compatible with your "
                f"system ({compatible_variant.variant_str}). Consider upgrading to that version by specifying "
                "the `version` argument."
            )
    except Exception:
        return ""

    return ""


def resolve_hub_cache_kernel(
    api: HfApi,
    repo_id: str,
    *,
    revision: str,
    backend: str | None,
) -> LocalKernel:
    """Resolve a kernel variant path from the local Hugging Face cache only.

    Used by `load_kernel` (which always operates on a pre-downloaded, locked
    kernel) and by the offline branch of `install_kernel`.
    """
    try:
        repo_path = Path(
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

    variants = get_variants_local(repo_path / "build")
    variant, status = resolve_variant(variants, backend)
    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}):\n\n{variants_trace_str(status)}"
        )

    variant_path = repo_path / "build" / variant.variant_str
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant path does not exist: `{variant_path}`")

    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    location = LocalKernel(variant_path=variant_path, metadata=metadata)

    return location


@runtime_checkable
class Resolver(Protocol):
    def resolve(
        self, *, api: HfApi, backend: str | None, kernel: KernelDependency
    ) -> LocalKernel | RemoteKernel | None:
        """Resolve a kernel dependency to a location.

        Returns `None` when the kernel cannot be resolved."""
        ...


@dataclass
class SequentialResolver:
    """Sequential kernel solver.

    This solver tries the embedded solvers sequentially until one succeeds."""

    resolvers: list[Resolver]

    def resolve(
        self, *, api: HfApi, backend: str | None, kernel: KernelDependency
    ) -> LocalKernel | RemoteKernel | None:
        for resolver in self.resolvers:
            location = resolver.resolve(api=api, backend=backend, kernel=kernel)
            if location is not None:
                return location

        return None


@dataclass
class HubResolver:
    """Hugging Face Hub kernel solver.

    This solver solves a kernel dependency by finding the kernel on the Hub."""

    trust_remote_code: bool | list[str]

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> RemoteKernel:
        # Check if the repo is trusted before downloading anything.
        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )

        # Resolve the revision that we need.
        revision = resolve_kernel_version(kernel, local_files_only=False)

        # Get the kernel metadata for the revision.
        return resolve_hub_kernel(kernel.repo_id, api=api, revision=revision, backend=backend)


@dataclass
class HubCacheResolver:
    """Hugging Face Hub cache kernel solver.

    This solver solves a kernel dependency from the local Hugging Face Hub
    cache. The kernel must already be present in the cache."""

    trust_remote_code: bool | list[str]

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel:
        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )

        # Resolve the revision that we need.
        revision = resolve_kernel_version(kernel, local_files_only=True)

        # Get the kernel metadata for the revision.
        return resolve_hub_cache_kernel(
            api,
            kernel.repo_id,
            revision=revision,
            backend=backend,
        )


def _locked_revision(kernel_locks: KernelLocks, kernel: KernelDependency) -> str:
    kernel_lock = kernel_locks.get(kernel, None)
    if kernel_lock is None:
        raise ValueError(
            f"Kernel `{kernel.repo_id}` is not locked. Please lock it with `kernels lock <project>` and then reinstall the project."
        )
    return kernel_lock.commit


@dataclass
class LockedHubResolver:
    """Lock file Hugging Face Hub kernel solver.

    This solver uses a lock file to solve kernel dependencies from the Hub."""

    kernel_locks: KernelLocks
    trust_remote_code: bool | list[str]

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> RemoteKernel:
        # Check if the repo is trusted before downloading anything.
        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=False,
            trust_remote_code=self.trust_remote_code,
        )

        revision = _locked_revision(self.kernel_locks, kernel)

        # Get the kernel metadata for the revision.
        return resolve_hub_kernel(kernel.repo_id, api=api, revision=revision, backend=backend)


@dataclass
class LockedHubCacheResolver:
    """Lock file Hugging Face Hub cache kernel solver.

    This solver uses a lock file to solve kernel dependencies from the local
    Hugging Face Hub cache. The kernel must already be present in the cache."""

    kernel_locks: KernelLocks
    trust_remote_code: bool | list[str]

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel:
        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=True,
            trust_remote_code=self.trust_remote_code,
        )

        revision = _locked_revision(self.kernel_locks, kernel)

        # Get the kernel metadata for the revision.
        return resolve_hub_cache_kernel(
            api,
            kernel.repo_id,
            revision=revision,
            backend=backend,
        )


@dataclass
class KernelPathsResolver:
    """Path-based kernel solver.

    This solver uses a kernel dependency to path mapping to solve kernel
    dependencies."""

    kernel_paths: KernelPaths

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel | None:
        repo_path = self.kernel_paths.get(kernel, None)
        if repo_path is None:
            return None

        return resolve_local_kernel(repo_path, backend=backend)


@dataclass
class RepoPathsResolver:
    """Path-based kernel solver.

    This solver uses a repo ID to path mapping to solve kernel dependencies."""

    local_kernels: dict[str, Path]

    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel | None:
        repo_path = self.local_kernels.get(kernel.repo_id, None)
        if repo_path is None:
            return None

        return resolve_local_kernel(repo_path, backend=backend)


@dataclass
class NoopResolver:
    def resolve(self, *, api: HfApi, backend: str | None, kernel: KernelDependency) -> LocalKernel | None:
        return None
