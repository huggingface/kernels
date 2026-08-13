from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Generic, Self, Tuple, TypeVar

from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.hf_api import HfApi
from kernels_data import KernelDependency, Metadata

from kernels._versions import resolve_kernel_version
from kernels.backends import _backend
from kernels.hf_hub import CACHE_DIR, _check_trust_remote_code
from kernels.importer import _import_from_path
from kernels.locking import KernelLock, KernelLocks
from kernels.python_deps import validate_dependencies
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


# Default state is `None`, to signal that we are not in a kernel
# loading context.
_KERNEL_DEPS: ContextVar[dict[str, ModuleType] | None] = ContextVar("_KERNEL_DEPS", default=None)


@dataclass(frozen=True)
class LocalKernel:
    """A kernel that can be loaded from a local path."""

    variant_path: Path

    variant: Variant | None

    def install(self, *, api: HfApi) -> Self:
        # Local kernels are already installed, so we just return self.
        return self


@dataclass(frozen=True)
class RemoteKernel:
    """A kernel that can be loaded from a remote path."""

    repo_id: str
    revision: str
    variant: Variant

    def install(self, *, api: HfApi) -> LocalKernel:

        allow_patterns = [f"build/{self.variant.variant_str}/*"]
        ignore_patterns = _BYTECODE_IGNORE_PATTERNS

        repo_path = Path(
            str(
                api.snapshot_download(
                    self.repo_id,
                    repo_type="kernel",
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    cache_dir=CACHE_DIR,
                    revision=self.revision,
                    local_files_only=False,
                )
            )
        )

        return LocalKernel(
            variant_path=repo_path / "build" / self.variant.variant_str,
            variant=self.variant,
        )


def get_kernel_metadata(
    repo_id: str,
    *,
    api: HfApi,
    backend: str | None,
    revision: str,
) -> Tuple[RemoteKernel, Path]:
    variants = get_variants(
        api,
        repo_id=repo_id,
        revision=revision,
    )
    variant, trace = resolve_variant(variants, backend)
    if variant is None:
        raise FileNotFoundError(
            f"Cannot find a build variant for this system in {repo_id} (revision: {revision}):\n\n{variants_trace_str(trace)}"
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

    location = RemoteKernel(repo_id=repo_id, revision=revision, variant=variant)

    return location, metadata_path


def _get_local_kernel_metadata(repo_path: Path, *, backend: str | None) -> Tuple[LocalKernel, Path]:
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

    metadata_path = variant_path / "metadata.json"
    location = LocalKernel(variant_path=variant_path, variant=variant)

    return location, metadata_path


def _get_offline_kernel_metadata(
    api: HfApi,
    repo_id: str,
    *,
    revision: str,
    backend: str | None,
) -> Tuple[LocalKernel, Path]:
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

    metadata_path = variant_path / "metadata.json"
    location = LocalKernel(variant_path=variant_path, variant=variant)

    return location, metadata_path


# Tree node type variable.
#
# `LocalKernel | RemoteKernel`: the initial tree.
# `LocalKernel`: the tree after installing kernels.
T = TypeVar("T", LocalKernel | RemoteKernel, LocalKernel)


@dataclass
class DepTreeNode(Generic[T]):
    metadata: Metadata
    location: T
    depends: dict[str, Self]

    def install(
        self,
        *,
        api: HfApi,
    ) -> "DepTreeNode[LocalKernel]":
        local_depends = {}
        for repo_id, node in self.depends.items():
            local_depends[repo_id] = node.install(api=api)

        return DepTreeNode(
            metadata=self.metadata,
            location=self.location.install(api=api),
            depends=local_depends,
        )

    def load(self) -> ModuleType:
        if isinstance(self.location, RemoteKernel):
            raise RuntimeError("Can only load installed kernels, run `install()` on the kernel dependency tree first.")

        deps = {repo_id: dep.load() for repo_id, dep in self.depends.items()}

        # TODO: add RepoInfo
        return _import_from_path(self.location.variant_path, None, kernel_deps=deps)

    def validate_dependencies(self) -> None:
        """Validate that the dependencies of this kernel are satisfied."""

        validate_dependencies(self.metadata.name.python_name, self.metadata.python_depends, _backend())

        for node in self.depends.values():
            node.validate_dependencies()


LoadCache = dict[KernelDependency, DepTreeNode]


def load_kernel_with_deps(
    *,
    api: HfApi,
    backend: str | None,
    local_kernels: dict[str, Path],
    kernel: KernelDependency,
    local_files_only: bool,
    kernel_locks: dict[str, KernelLock] | None,
    trust_remote_code: bool | list[str],
) -> ModuleType:
    tree = resolve_kernel_tree(
        api=api,
        backend=backend,
        local_kernels=local_kernels,
        kernel=kernel,
        kernel_locks=kernel_locks,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tree.validate_dependencies()
    tree_only_local = tree.install(api=api)
    return tree_only_local.load()


def resolve_kernel_tree(
    api: HfApi,
    kernel: KernelDependency,
    backend: str | None,
    kernel_locks: KernelLocks | None,
    local_files_only: bool,
    local_kernels: dict[str, Path],
    trust_remote_code: bool | list[str],
    seen: set[KernelDependency] | None = None,
) -> DepTreeNode[LocalKernel | RemoteKernel]:
    """
    Recursively solve kernel dependencies.

    Constructs a tree where nodes encode kernel information (e.g. location and
    metadata) and edges kernel-kernel dependendies.
    """
    if seen is None:
        seen = set()

    print(kernel)
    print(kernel_locks)

    # Check for cycles.
    if kernel in seen:
        raise ValueError(f"Cyclic kernel dependency detected: {kernel.repo_id}")
    seen.add(kernel)

    location: LocalKernel | RemoteKernel
    metadata_path: Path
    kernel_lock = None

    if kernel.repo_id in local_kernels:
        # Shortcut for kernels with local overrides.
        repo_path = local_kernels[kernel.repo_id]
        location, metadata_path = _get_local_kernel_metadata(repo_path, backend=backend)
    else:
        # Check if the repo is trusted before downloading anything.
        _check_trust_remote_code(
            repo_id=kernel.repo_id,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )

        # Resolve the revision that we need.
        if kernel_locks is not None:
            # If kernel locks are provided, we use the revision from the
            # locks. We also require *all* kernels to be locked.
            kernel_lock = kernel_locks.locks.get(kernel, None)
            if kernel_lock is None:
                raise ValueError(
                    f"Kernel `{kernel.repo_id}` is not locked. Please lock it with `kernels lock <project>` and then reinstall the project."
                )
            revision = kernel_lock.revision
        else:
            revision = resolve_kernel_version(kernel, local_files_only=local_files_only)

        # Get the kernel metadata for the revision.
        if local_files_only:
            location, metadata_path = _get_offline_kernel_metadata(
                api,
                kernel.repo_id,
                revision=revision,
                backend=backend,
            )
        else:
            location, metadata_path = get_kernel_metadata(kernel.repo_id, api=api, revision=revision, backend=backend)

            # if kernel_locks is not None:
            # if isinstance(location, LocalKernel) and location.variant is None:
            #    raise ValueError("Cannot determine variant for a local kernel.")
            # kernel_lock = kernel_locks.get(kernel.repo_id, None)
            # assert kernel_lock is not None
            # kernel_locks = kernel_lock.depends.get(location.variant.variant_str, None)
            # if kernel_locks is None:
            #    raise ValueError(
            #        f"Kernel `{kernel.repo_id}` does not have a lock for variant `{location.variant.variant_str}`. Please lock it with `kernels lock <project>` and then reinstall the project."
            #    )

    metadata = Metadata.read_from_file(metadata_path)

    kernel_deps = {}
    print("depends", kernel_lock, kernel_lock.depends)

    # Recurse into dependencies.
    for dep in metadata.kernel_depends:
        kernel_deps[dep.repo_id] = resolve_kernel_tree(
            api=api,
            backend=backend,
            seen=seen,
            kernel_locks=(kernel_lock.depends if kernel_lock is not None else None),
            local_files_only=local_files_only,
            local_kernels=local_kernels,
            kernel=dep,
            trust_remote_code=trust_remote_code,
        )

    seen.remove(kernel)

    return DepTreeNode(
        metadata=metadata,
        location=location,
        depends=kernel_deps,
    )


def use_kernel_deps(deps: dict[str, ModuleType]):
    class ContextManager:
        def __enter__(self):
            self.token = _KERNEL_DEPS.set(deps)

        def __exit__(self, exc_type, exc_value, traceback):
            _KERNEL_DEPS.reset(self.token)

    return ContextManager()


def get_kernel_dep(repo_id: str) -> ModuleType:
    deps = _KERNEL_DEPS.get()
    if deps is None:
        raise RuntimeError("`get_kernel_dep` only works during kernel loading.")

    module = deps.get(repo_id)
    if module is None:
        raise RuntimeError(
            f"Dependency '{repo_id}' not found in kernel dependencies. Ensure the dependency is added to `kernel-deps` in build.toml."
        )

    return module
