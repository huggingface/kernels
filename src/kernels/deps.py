import importlib.util
import json
from pathlib import Path
from typing import Dict, List

try:
    with open(Path(__file__).parent / "python_depends.json", "r") as f:
        DEPENDENCY_DATA: Dict = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        "Cannot load dependency data, is `kernels` correctly installed?"
    )


def validate_dependencies(dependencies: List[str], backend: str):
    """
    Validate a list of dependencies to ensure they are installed.

    Args:
        dependencies (`List[str]`): A list of dependency strings to validate.
        backend (`str`): The backend to validate dependencies for.
    """
    general_deps = DEPENDENCY_DATA.get("general", {})
    backend_deps = DEPENDENCY_DATA.get("backends", {}).get(backend, {})

    # Validate each dependency
    for dependency in dependencies:
        # Look up dependency in general dependencies first, then backend-specific
        if dependency in general_deps:
            python_packages = general_deps[dependency].get("python", [])
        elif dependency in backend_deps:
            python_packages = backend_deps[dependency].get("python", [])
        else:
            # Dependency not found in general or backend-specific dependencies
            raise ValueError(f"Invalid dependency: {dependency}")

        # Check if each python package is installed
        for python_package in python_packages:
            # Convert package name to module name (replace - with _)
            module_name = python_package.replace("-", "_")
            if importlib.util.find_spec(module_name) is None:
                raise ImportError(
                    f"Kernel requires Python dependency `{python_package}`. Please install with: pip install {python_package}"
                )
