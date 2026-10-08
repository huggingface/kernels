"""Inspect a built kernel's public functions and layer classes into symbols.json.

Run this in the build's Python environment, with the kernel's runtime
dependencies and the kernels package available. The build must include its
metadata.json. Importing a kernel executes its initialization code.
The inspection itself does not call exported functions or instantiate classes.
"""

import argparse
import importlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


def describe_signature(obj: Any) -> dict[str, Any] | None:
    """Serialize inspect.Signature without evaluating string annotations."""
    try:
        signature = inspect.signature(obj, eval_str=False)
    except (TypeError, ValueError):
        # Some extension functions and callable objects expose no signature.
        return None

    return {
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "default": (
                    None
                    if parameter.default is inspect.Parameter.empty
                    else repr(parameter.default)
                ),
                "annotation": (
                    None
                    if parameter.annotation is inspect.Parameter.empty
                    else inspect.formatannotation(parameter.annotation)
                ),
            }
            for parameter in signature.parameters.values()
        ],
        "return_annotation": (
            None
            if signature.return_annotation is inspect.Signature.empty
            else inspect.formatannotation(signature.return_annotation)
        ),
    }


def describe_symbol(name: str, obj: Any) -> dict[str, Any]:
    """Describe a public function or layer class using inspect terminology."""
    symbol = {
        "name": name,
        "qualname": obj.__qualname__,
        "module": obj.__module__,
        "kind": "class" if inspect.isclass(obj) else "function",
        "doc": inspect.getdoc(obj),
        "signature": describe_signature(obj),
    }
    if inspect.isclass(obj):
        symbol["attributes"] = {
            attribute: value
            if isinstance(value := getattr(obj, attribute, None), bool)
            else None
            for attribute in ("has_backward", "can_torch_compile")
        }
    return symbol


def describe_exports(
    module: ModuleType, predicate: Callable[[Any], bool]
) -> list[dict[str, Any]]:
    """Select public exports of the requested kind, preserving __all__ order."""
    exports = getattr(module, "__all__", ())
    if not isinstance(exports, (list, tuple)) or not all(
        isinstance(export, str) for export in exports
    ):
        raise ValueError(
            f"{module.__name__}.__all__ must be a list or tuple of strings"
        )
    symbols = []
    for name in dict.fromkeys(exports):
        if name.startswith("_") or name == "layers":
            continue
        try:
            obj = getattr(module, name)
        except AttributeError as error:
            raise ValueError(
                f"{module.__name__}.__all__ names a missing symbol: {name!r}"
            ) from error
        if not predicate(obj):
            continue
        if "_private_for_testing" in obj.__module__.split("."):
            continue
        symbols.append(describe_symbol(name, obj))
    return symbols


def generate_symbols(module: ModuleType) -> dict[str, Any]:
    """Describe top-level functions and optional layers, each scoped by __all__.

    As in describe-api, layers need not appear in the package's __all__.
    Only Python functions (including async functions) and layer classes are
    included; other exported modules, data, and callable objects are ignored.
    """
    functions = describe_exports(module, inspect.isfunction)
    layers = getattr(module, "layers", None)
    if layers is None and hasattr(module, "__path__"):
        # The package need not import its optional layers submodule itself.
        layers_name = f"{module.__name__}.layers"
        try:
            layers = importlib.import_module(layers_name)
        except ModuleNotFoundError as error:
            if error.name != layers_name:
                raise
    return {
        "schema_version": 1,
        "module": module.__name__,
        "functions": functions,
        "layers": describe_exports(layers, inspect.isclass)
        if inspect.ismodule(layers)
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kernel_dir",
        type=Path,
        help="Local kernel repository or build variant containing metadata.json",
    )
    parser.add_argument(
        "--backend", help="Backend to select when loading a local kernel repository"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Destination symbols.json"
    )
    args = parser.parse_args()

    from kernels import get_local_kernel

    module = get_local_kernel(args.kernel_dir, backend=args.backend)
    # Finish inspection and serialization before touching an existing output file.
    content = json.dumps(generate_symbols(module), indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
