from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, TypeVar

from huggingface_hub.hf_api import HfApi
from kernels_data import KernelDependency, KernelVersion

from kernels.archs import _check_arch_incompatibility
from kernels.backends import _backend
from kernels.hf_hub import RepoInfo
from kernels.importer import _import_from_path
from kernels.python_deps import validate_dependencies
from kernels.resolver import (
    LocalKernel,
    RemoteKernel,
    Resolver,
)

# Default state is `None`, to signal that we are not in a kernel
# loading context.
_KERNEL_DEPS: ContextVar[dict[str, ModuleType] | None] = ContextVar("_KERNEL_DEPS", default=None)


# Tree node type variable.
#
# `LocalKernel | RemoteKernel`: the initial tree.
# `LocalKernel`: the tree after installing kernels.
T = TypeVar("T", LocalKernel | RemoteKernel, LocalKernel)


@dataclass
class DepTreeNode(Generic[T]):
    """Node in a kernel dependency tree.

    The content of the node is the location of the kernel (`location`). The
    outgoing edges (`deps`) are dependencies of the kernel."""

    location: T
    deps: dict[str, "DepTreeNode[T]"]

    def install(self, *, api: HfApi) -> "DepTreeNode[LocalKernel]":
        """Install the kernel and its dependencies.

        This will download the kernel to the Hub cache if it is a remote
        kernel. Otherwise, installation is a no-op."""
        local_deps = {}
        for repo_id, node in self.deps.items():
            local_deps[repo_id] = node.install(api=api)

        return DepTreeNode(
            location=self.location.install(api=api),
            deps=local_deps,
        )

    def load(self) -> ModuleType:
        if isinstance(self.location, RemoteKernel):
            raise RuntimeError("Can only load installed kernels, run `install()` on the kernel dependency tree first.")

        deps = {repo_id: dep.load() for repo_id, dep in self.deps.items()}

        repo_info = (
            RepoInfo(
                repo_id=self.location.origin.repo_id,
                revision=self.location.origin.revision,
            )
            if self.location.origin
            else None
        )

        return _import_from_path(self.location.variant_path, repo_info=repo_info, deps=deps)

    def check_archs(self) -> None:
        """Check that this kernel and its dependencies support the current device.

        Raises `RuntimeError` when a kernel build does not support the
        architecture (e.g. CUDA compute capability) of the current device."""

        _check_arch_incompatibility(self.location.metadata, self.location.variant_str)

        for node in self.deps.values():
            node.check_archs()

    def validate_dependencies(self) -> None:
        """Validate that the dependencies of this kernel are satisfied."""

        validate_dependencies(
            self.location.metadata.name.python_name,
            self.location.metadata.python_depends,
            _backend(),
        )

        for node in self.deps.values():
            node.validate_dependencies()


LoadCache = dict[KernelDependency, DepTreeNode]


def resolve_kernel_tree(
    api: HfApi,
    kernel: KernelDependency,
    backend: str | None,
    resolver: Resolver | None,
    seen: set[KernelDependency] | None = None,
) -> DepTreeNode[LocalKernel | RemoteKernel]:
    """
    Recursively solve kernel dependencies.

    Constructs a tree where nodes encode kernel information (e.g. location and
    metadata) and edges kernel-kernel dependendies.
    """
    if seen is None:
        seen = set()

    # Check for cycles.
    if kernel in seen:
        raise ValueError(f"Cyclic kernel dependency detected: {kernel.repo_id}")
    seen.add(kernel)

    location: LocalKernel | RemoteKernel | None

    location = resolver.resolve(api=api, backend=backend, kernel=kernel) if resolver else None

    if location is None:
        match kernel.version:
            case KernelVersion.Version(version=version):
                version_str = f"version: {version}"
            case KernelVersion.Revision(revision=revision):
                version_str = f"revision: {revision}"
        raise ValueError(f"Could not resolve kernel: {kernel.repo_id} ({version_str})")

    # Recurse into dependencies.
    kernel_deps = {}
    for dep in location.metadata.kernel_depends:
        kernel_deps[dep.repo_id] = resolve_kernel_tree(
            api=api,
            backend=backend,
            seen=seen,
            kernel=dep,
            resolver=resolver,
        )

    seen.remove(kernel)

    return DepTreeNode(
        location=location,
        deps=kernel_deps,
    )


def use_kernel_deps(deps: dict[str, ModuleType]):
    class ContextManager:
        def __enter__(self):
            self.token = _KERNEL_DEPS.set(deps)

        def __exit__(self, exc_type, exc_value, traceback):
            _KERNEL_DEPS.reset(self.token)

    return ContextManager()


def get_kernel_dep(repo_id: str) -> ModuleType:
    """Get a kernel dependency.

    This function can be used by a kernel to get one of its dependencies.
    The exact version/revision of the dependency must be encoded in the
    kernel metadata. This function can only be called during kernel loading.

    Args:
        repo_id (`str`):
            The Hub kernel repository containing the dependency.
    """
    deps = _KERNEL_DEPS.get()
    if deps is None:
        raise RuntimeError("`get_kernel_dep` only works during kernel loading.")

    module = deps.get(repo_id)
    if module is None:
        raise RuntimeError(
            f"Dependency '{repo_id}' not found in kernel dependencies. Ensure the dependency is added to `kernel-deps` in build.toml."
        )

    return module
