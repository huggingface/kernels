import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from huggingface_hub.dataclasses import strict

from kernels.backends import Backend


@strict
@dataclass
class PythonPackage:
    pkg: str
    import_name: Optional[str] = None

    @staticmethod
    def from_dict(data: dict) -> "PythonPackage":
        return PythonPackage(
            pkg=data["pkg"],
            import_name=data.get("import"),
        )


@strict
@dataclass
class DependencyInfo:
    nix: list
    python: list

    @staticmethod
    def from_dict(data: dict) -> "DependencyInfo":
        return DependencyInfo(
            nix=data.get("nix", []),
            python=[PythonPackage.from_dict(p) for p in data.get("python", [])],
        )


@strict
@dataclass
class DependencyData:
    general: dict = field(default_factory=dict)
    backends: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(data: dict) -> "DependencyData":
        general = {
            name: DependencyInfo.from_dict(info)
            for name, info in data.get("general", {}).items()
        }
        backends = {
            backend_name: {
                name: DependencyInfo.from_dict(info) for name, info in deps.items()
            }
            for backend_name, deps in data.get("backends", {}).items()
        }
        return DependencyData(general=general, backends=backends)


try:
    with open(Path(__file__).parent / "python_depends.json", "r") as f:
        _DEPENDENCY_DATA = DependencyData.from_dict(json.load(f))
except FileNotFoundError:
    raise FileNotFoundError(
        "Cannot load dependency data, is `kernels` correctly installed?"
    )


def validate_dependencies(
    kernel_module_name: str, dependencies: list[str], backend: Backend
):
    """
    Validate a list of dependencies to ensure they are installed.

    Args:
        dependencies (`list[str]`): A list of dependency strings to validate.
        backend (`str`): The backend to validate dependencies for.
    """
    general_deps = _DEPENDENCY_DATA.general
    backend_deps = _DEPENDENCY_DATA.backends.get(backend.name, {})

    # Validate each dependency
    for dependency in dependencies:
        # Look up dependency in general dependencies first, then backend-specific
        if dependency in general_deps:
            dep_info = general_deps[dependency]
        elif dependency in backend_deps:
            dep_info = backend_deps[dependency]
        else:
            # Dependency not found in general or backend-specific dependencies
            raise ValueError(
                f"Kernel module `{kernel_module_name}` uses unsupported kernel dependency: {dependency}"
            )

        # Check if each python package is installed
        for python_package in dep_info.python:
            pkg_name = python_package.pkg
            # Assertion because this should not happen and is a bug.
            assert (
                pkg_name is not None
            ), f"Invalid dependency data for `{dependency}`: missing `pkg` field."

            module_name = python_package.import_name
            if module_name is None:
                # These are typically packages that do not provide any Python
                # code, but get installed to Python's library dirctory. E.g.
                # OneAPI.
                continue

            if importlib.util.find_spec(module_name) is None:
                raise ImportError(
                    f"Kernel module `{kernel_module_name}` requires Python dependency `{pkg_name}`. Please install with: pip install {pkg_name}"
                )
