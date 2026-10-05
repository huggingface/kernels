import inspect
import json

from kernels import get_kernel


def print_kernel_api(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    json_output: bool = False,
):
    """Load a kernel and print its exported functions and layers."""
    kernel = get_kernel(repo_id, revision=revision, version=version)
    info = {
        "repo_id": repo_id,
        "revision": revision,
        "version": version,
        "functions": [name for name, _ in inspect.getmembers(kernel, inspect.isfunction)],
        "layers": [name for name, _ in inspect.getmembers(getattr(kernel, "layers", None), inspect.isclass)],
    }

    if json_output:
        print(json.dumps(info, indent=2))
    else:
        _print_human(info)


def _print_human(info: dict):
    selection = info["revision"] if info["revision"] is not None else f"v{info['version']}"
    print(f"Repository: {info['repo_id']}")
    print(f"Revision: {selection}")
    print("Functions:")
    for function in info["functions"]:
        print(f"  {function}")
    print("Layers:")
    for layer in info["layers"]:
        print(f"  {layer}")
