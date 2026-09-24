import importlib
import importlib.machinery
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from kernels._rust import Metadata
from kernels.hf_hub import RepoInfo


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


class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """
    Source loader that always compiles from the `.py` file.

    Bytecode is excluded from the kernel digest, since the interpreter
    writes it after the first import. So bytecode cannot be trusted and
    should never be executed in place of the verified source.
    """

    def get_code(self, fullname):
        source_path = self.get_filename(fullname)
        return self.source_to_code(self.get_data(source_path), source_path)

    def set_data(self, path, data, *, _mode=0o666):
        # Do not write bytecode that is never read.
        pass


def _register_source_only_finders(module_dir: Path):
    """
    Register finders for every directory in the kernel module, so that
    submodules are also loaded from source. Sourceless `.pyc` files are
    not importable at all.
    """
    loaders = [
        (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES),
        (_SourceOnlyLoader, importlib.machinery.SOURCE_SUFFIXES),
    ]
    for root, dirs, _ in os.walk(module_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        sys.path_importer_cache[root] = importlib.machinery.FileFinder(root, *loaders)


def _import_from_path(
    variant_path: Path,
    deps: dict[str, ModuleType],
    repo_info: RepoInfo | None = None,
) -> ModuleType:
    if (loaded_kernel := _loaded_kernels.get(variant_path)) is not None:
        return loaded_kernel.module

    metadata = Metadata.read_from_file(variant_path / "metadata.json")
    module_name = metadata.name.python_name

    file_path = variant_path / "__init__.py"
    if not file_path.exists():
        file_path = variant_path / module_name / "__init__.py"
    if not file_path.exists():
        raise FileNotFoundError(f"No kernel module found at: `{variant_path}`")

    _register_source_only_finders(file_path.parent)
    spec = importlib.util.spec_from_file_location(
        metadata.id, file_path, loader=_SourceOnlyLoader(metadata.id, str(file_path))
    )
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

    _loaded_kernels[variant_path] = LoadedKernel(
        metadata=metadata,
        module=module,
        repo_info=repo_info,
    )
    return module
