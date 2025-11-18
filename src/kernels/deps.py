import importlib.util
from typing import List, Set

allowed_dependencies: Set[str] = {
    "einops",
    "nvidia-cutlass-dsl",
}


def validate_dependencies(dependencies: List[str]):
    """
    Validate a list of dependencies to ensure they are installed.

    Args:
        dependencies (`List[str]`): A list of dependency strings.
    """
    for dependency in dependencies:
        if dependency not in allowed_dependencies:
            allowed = ", ".join(sorted(allowed_dependencies))
            raise ValueError(
                f"Invalid dependency: {dependency}, allowed dependencies: {allowed}"
            )

        if importlib.util.find_spec(dependency.replace("-", "_")) is None:
            raise ImportError(
                f"Kernel requires dependency `{dependency}`. Please install with: pip install {dependency}"
            )
