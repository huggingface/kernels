import importlib.util
import json
from pathlib import Path

from kernels.backends import Backend

try:
    with open(Path(__file__).parent / "python_depends.json", "r") as f:
        DEPENDENCY_DATA: dict = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "Cannot load dependency data, is `kernels` correctly installed?"
    )


def validate_dependencies(dependencies: list[str], backend: Backend):
    """
    Validate a list of dependencies to ensure they are installed.

    Args:
        dependencies (`list[str]`): A list of dependency strings to validate.
        backend (`str`): The backend to validate dependencies for.
    """
    general_deps = DEPENDENCY_DATA.get("general", {})
    backend_deps = DEPENDENCY_DATA.get("backends", {}).get(backend.name, {})

    # Validate each dependency
    for dependency in dependencies:
        # Look up dependency in general dependencies first, then backend-specific
        if dependency in general_deps:
            python_packages = general_deps[dependency].get("python", [])
        elif dependency in backend_deps:
            python_packages = backend_deps[dependency].get("python", [])
        else:
            # Dependency not found in general or backend-specific dependencies
            raise ValueError(f"Unsupported kernel dependency: {dependency}")

        # Check if each python package is installed
        for python_package in python_packages:
            # Convert package name to module name (replace - with _)
            pkg_name = python_package.get("pkg")
            # Assertion because this should not happen and is a bug.
            assert (
                pkg_name is not None
            ), f"Invalid dependency data for `{dependency}`: missing `pkg` field."

            module_name = python_package.get("import")
            if module_name is None:
                print("wut?")
                # These are typically packages that do not provide any Python
                # code, but get installed to Python's library dirctory. E.g.
                # OneAPI.
                continue

            if importlib.util.find_spec(module_name) is None:
                raise ImportError(
                    f"Kernel requires Python dependency `{pkg_name}`. Please install with: pip install {pkg_name}"
                )
