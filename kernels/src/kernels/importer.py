import importlib
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from kernels._rust import DigestValidationError, Metadata
from kernels.hf_hub import RepoInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedKernel:
    """
    This dataclass provides information about a loaded kernel:

    - `metadata` (`Metadata`): kernel metadata.
    - `module` (`ModuleType`): the imported kernel module.
    - `repo_info` (`kernels.hf_hub.RepoInfo | None`): populated whenever the
      Hub repository the kernel came from is known.

    The metadata includes the following properties that describe a kernel:

    - `id` (`str`): kernel identifier that is unique to the kernel version + backend.
    - `name` (`str`): the name of the kernel.
    - `version` (`int`): the version of the kernel.
    - `kernels_minver` (`Version | None`): the minimum `kernels` library
      version required to load the kernel.
    - `license` (`str`): the license of the kernel.
    - `upstream` (`str | None`): the original upstream repository of the kernel.
    - `source` (`str | None`): the kernel-builder formatted source repository.
    - `python_depends` (`list[str]`): required Python dependencies.
    - `backend`: information about the kernel's backend.
    """

    metadata: Metadata
    module: ModuleType
    repo_info: RepoInfo | None


_loaded_kernels: dict[Path, LoadedKernel] = {}

# Metadata of loaded kernels by their ids.
_loaded_kernel_metadata: dict[str, Metadata] = {}

# Serializes kernel imports, so that a kernel is executed only once and other
# threads never see a partially initialized kernel module. Reentrant, so that a
# kernel that loads another kernel while it is imported does not deadlock.
_import_lock = threading.RLock()


def get_loaded_kernels() -> list[LoadedKernel]:
    """
    Return a snapshot of every kernel that has been loaded into the current process.

    The returned list is a new list; mutating it does not affect the registry.

    Returns:
        `list[LoadedKernel]`: One [`LoadedKernel`] per distinct kernel variant path
        loaded in this process.

    Example:
        ```python
        from kernels import get_kernel, get_loaded_kernels

        get_kernel("kernels-community/activation", version=1)
        for loaded in get_loaded_kernels():
            print(loaded.metadata.name, loaded.repo_info)
        ```
    """
    return list(_loaded_kernels.values())


def _check_same_build(variant_path: Path, metadata: Metadata) -> None:
    """Check that the kernel at `variant_path` is the same build as the loaded
    kernel with the same id.

    If there are two different kernels with the same kernel id, one (or both)
    of the kernels violates the unique kernel id requirement.

    Raises `RuntimeError` when the builds differ.
    """
    reference_metadata = _loaded_kernel_metadata.get(metadata.id)
    assert reference_metadata is not None, (
        f"Kernel '{metadata.id}' is in `sys.modules`, but its metadata was not recorded"
    )

    reference_digest = reference_metadata.digest
    digest = metadata.digest

    if reference_digest is None and digest is None:
        logger.debug(
            f"Cannot compare kernel '{metadata.id}' at `{variant_path}` with the loaded build: neither has a digest"
        )
        return

    conflict = (
        f"Kernel '{metadata.name}' at `{variant_path}` has the same id '{metadata.id}' as a kernel "
        "that is already loaded, but it is a different build"
    )
    hint = "Was the kernel modified without rebuilding it with `kernel-builder`?"

    if reference_digest is None:
        raise RuntimeError(f"{conflict}: this build has a digest, but the loaded build does not. {hint}")
    if digest is None:
        raise RuntimeError(f"{conflict}: the loaded build has a digest, but this build does not. {hint}")

    try:
        reference_digest.validate(digest)
    except DigestValidationError as e:
        violations = "\n".join(str(violation) for violation in e.violations)
        raise RuntimeError(f"{conflict}. {hint}\nDifferences with the loaded build:\n{violations}") from e


def _import_from_path(
    variant_path: Path,
    deps: dict[str, ModuleType],
    repo_info: RepoInfo | None = None,
) -> ModuleType:
    metadata = Metadata.read_from_file(variant_path / "metadata.json")

    with _import_lock:
        # Kernel ids are unique per build: if this build was already imported
        # reuse it instead of executing it again.
        if (module := sys.modules.get(metadata.id)) is None:
            module = _import_from_path_uncached(variant_path, metadata, deps, repo_info)
        else:
            logger.debug(f"Kernel already loaded, skipping: {metadata.id}")
            _check_same_build(variant_path, metadata)

        _loaded_kernels[variant_path] = LoadedKernel(
            metadata=metadata,
            module=module,
            repo_info=repo_info,
        )
        return module


def _import_from_path_uncached(
    variant_path: Path,
    metadata: Metadata,
    deps: dict[str, ModuleType],
    repo_info: RepoInfo | None,
) -> ModuleType:
    """Import the kernel at `variant_path`, without reusing an imported kernel with the same id.

    Must be called with `_import_lock` held.
    """
    module_name = metadata.name.python_name

    file_path = variant_path / "__init__.py"
    if not file_path.exists():
        file_path = variant_path / module_name / "__init__.py"
    if not file_path.exists():
        raise FileNotFoundError(f"No kernel module found at: `{variant_path}`")

    spec = importlib.util.spec_from_file_location(metadata.id, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Cannot load module {module_name} from spec")
    sys.modules[metadata.id] = module

    # Avoid an import cycle.
    from kernels.deps import use_kernel_deps

    try:
        with use_kernel_deps(deps):
            spec.loader.exec_module(module)  # type: ignore
    except Exception as e:
        # Remove the partially initialized module, so that a retry
        # imports from scratch.
        sys.modules.pop(metadata.id, None)
        if hasattr(e, "add_note"):
            origin = f"({repo_info.repo_id}, revision: {repo_info.revision})" if repo_info else ""
            e.add_note(f"while importing kernel '{metadata.name}', variant '{variant_path.name}' {origin}")
        raise

    _loaded_kernel_metadata[metadata.id] = metadata
    return module
