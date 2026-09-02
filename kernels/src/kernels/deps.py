from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, Protocol, TypeVar

from huggingface_hub.hf_api import HfApi
from kernels_data import KernelDependency, KernelVersion, Version
from packaging.version import InvalidVersion, parse

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

    def validate(
        self: "DepTreeNode[LocalKernel | RemoteKernel]",
        validator: "Validator",
    ) -> None:
        """Validate this kernel and its dependencies with the given validator."""

        validator.validate(tree=self)

        for node in self.deps.values():
            node.validate(validator)


class Validator(Protocol):
    """Validator for a node in a resolved kernel dependency tree."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None: ...


class DependencyValidator:
    """Validate the Python dependencies of a kernel tree node."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        validate_dependencies(
            tree.location.metadata.name.python_name,
            tree.location.metadata.python_depends,
            _backend(),
        )


class ArchValidator:
    """Validate architecture compatibility for a kernel tree node."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        _check_arch_incompatibility(tree.location.metadata, tree.location.variant_str)


def _installed_version() -> Version | None:
    """The installed `kernels` version as a numeric version.

    Pre-release and development suffixes are stripped, since kernel metadata
    only records release versions. Without stripping, a development version
    like `0.17.0.dev0` would compare as older than the `0.17.0` it implements.

    Returns `None` if the installed version is not a PEP 440 version, which
    can happen for versions derived from a VCS checkout. Such a version cannot
    be compared, and this check must never make a kernel fail to load.
    """
    # Avoid an import cycle.
    from kernels import __version__

    try:
        release = parse(__version__).release
    except InvalidVersion:
        return None

    # packaging < 22 returns a LegacyVersion with `release = None` for
    # non-PEP 440 versions instead of raising `InvalidVersion`.
    if release is None:
        return None

    return Version.from_str(".".join(str(part) for part in release))


class MinverValidator:
    """Validate that the installed `kernels` library meets the minimum version
    required by a kernel."""

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        metadata = tree.location.metadata
        minver = metadata.kernels_minver
        if minver is None:
            return

        installed = _installed_version()
        if installed is not None and installed < minver:
            # Report the verbatim installed version, not the normalized one used
            # for comparison, so the message matches what `pip show` reports.
            # Avoid an import cycle.
            from kernels import __version__

            raise RuntimeError(
                f"Kernel '{metadata.name}' variant '{tree.location.variant_str}' requires "
                f"kernels>={minver}, but version {__version__} is installed. "
                "Upgrade with: pip install --upgrade kernels"
            )


@dataclass
class AllValidator:
    """Apply multiple validators to a kernel dependency tree."""

    validators: list[Validator]

    def validate(self, *, tree: DepTreeNode[LocalKernel | RemoteKernel]) -> None:
        for validator in self.validators:
            validator.validate(tree=tree)


def default_validators() -> list[Validator]:
    """The validators that are applied to every kernel dependency tree."""
    return [DependencyValidator(), MinverValidator()]


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
