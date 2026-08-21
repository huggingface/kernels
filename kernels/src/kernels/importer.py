import importlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from kernels_data import Metadata

from kernels.hf_hub import RepoInfo


@dataclass(frozen=True)
class LoadedKernel:
    """
    This dataclass provides information about a loaded kernel:

    - `metadata` (`Metadata`): kernel metadata.
    - `module` (`ModuleType`): the imported kernel module.
    - `repo_info` (`kernels.hf_hub.RepoInfo | None`): populated only for
      kernels loaded via `get_kernel`. Loaders that work from a local path
      (`get_local_kernel`) or a lockfile (`get_locked_kernel`, `load_kernel`)
      leave this as `None`.

    The metadata includes the following properties that describe a kernel:

    - `id` (`str`): kernel identifier that is unique to the kernel version + backend.
    - `name` (`str`): the name of the kernel.
    - `version` (`int`): the version of the kernel.
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


def _warn_if_dirty(metadata: Metadata, variant_str: str) -> None:
    """Warn when a kernel variant was built from a dirty git tree.

    A dirty build was produced from a working tree with uncommitted changes,
    so its git SHA does not fully identify the sources it was built from and
    the build may not be reproducible.
    """
    provenance = metadata.provenance
    if provenance is None or not provenance.dirty:
        return

    dirty_sources = []
    if provenance.kernel is not None and provenance.kernel.dirty:
        dirty_sources.append("kernel source")
    builder_git = provenance.kernel_builder.git
    if builder_git is not None and builder_git.dirty:
        dirty_sources.append("kernel-builder")

    warnings.warn(
        f"Kernel '{metadata.name}' variant '{variant_str}' was built from a dirty "
        f"git tree ({', '.join(dirty_sources)} had uncommitted changes). Its "
        "recorded git revision does not fully identify the sources it was built "
        "from, so the build may not be reproducible.",
        stacklevel=3,
    )


def _import_from_path(
    variant_path: Path,
    deps: dict[str, ModuleType],
    repo_info: RepoInfo | None = None,
) -> ModuleType:
    if (loaded_kernel := _loaded_kernels.get(variant_path)) is not None:
        return loaded_kernel.module

    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    _warn_if_dirty(metadata, variant_path.name)
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
    spec.loader.exec_module(module)  # type: ignore

    _loaded_kernels[variant_path] = LoadedKernel(
        metadata=metadata,
        module=module,
        repo_info=repo_info,
    )
    return module
