import inspect
import re
import sys
from types import ModuleType

import yaml

from ._vendored.convert_rst_to_mdx import convert_rst_docstring_to_mdx
from .utils import get_kernel

_RE_PARAMETERS = re.compile(
    r"<parameters>(((?!<parameters>).)*)</parameters>", re.DOTALL
)
_RE_RETURNS = re.compile(r"<returns>(((?!<returns>).)*)</returns>", re.DOTALL)
_RE_RETURNTYPE = re.compile(
    r"<returntype>(((?!<returntype>).)*)</returntype>", re.DOTALL
)


def _extract_description_before_tags(docstring_mdx: str) -> str:
    """Extract the description part of a docstring before any tags."""
    params_pos = docstring_mdx.find("<parameters>")
    returns_pos = docstring_mdx.find("<returns>")
    returntype_pos = docstring_mdx.find("<returntype>")
    positions = [pos for pos in [params_pos, returns_pos, returntype_pos] if pos != -1]

    if positions:
        first_tag_pos = min(positions)
        return docstring_mdx[:first_tag_pos].strip()
    else:
        return docstring_mdx.strip()


def _print_parameters_section(docstring_mdx: str, *, header_level: int) -> None:
    """Print the parameters section from a docstring."""
    matches = _RE_PARAMETERS.findall(docstring_mdx)
    if matches:
        header = "#" * header_level
        print(f"\n{header} Parameters")
        for match in matches:
            print(f"\n{match[0].strip()}")


def _print_returns_section(
    docstring_mdx: str, *, context_name: str, header_level: int
) -> None:
    """Print the returns section from a docstring."""
    return_matches = _RE_RETURNS.findall(docstring_mdx)
    returntype_matches = _RE_RETURNTYPE.findall(docstring_mdx)

    if return_matches or returntype_matches:
        header = "#" * header_level
        print(f"\n{header} Returns")

        if returntype_matches:
            if len(returntype_matches) > 1:
                raise ValueError(
                    f"More than one <returntype> tag found in docstring for {context_name}"
                )
            print(f"\n**Type**: {returntype_matches[0][0].strip()}")

        if return_matches:
            for match in return_matches:
                print(f"\n{match[0].strip()}")


def _get_docstring(obj, use_dict_check: bool = False) -> str:
    """Get docstring from an object, with fallback to default message."""
    # Check whether the class/method itself has docs and not just
    # the superclass.
    if use_dict_check:
        has_doc = obj.__dict__.get("__doc__", None) is not None
    else:
        has_doc = getattr(obj, "__doc__", None) is not None

    # We use inspect.getdoc because it does normalization.
    doc = inspect.getdoc(obj)

    return doc if has_doc and doc is not None else "No documentation available."


def _process_and_print_docstring(
    docstring: str, *, kernel_name: str, context_name: str, header_level: int
) -> None:
    """Convert docstring to MDX and print description, parameters, and returns sections."""
    docstring_mdx = convert_rst_docstring_to_mdx(
        docstring, page_info={"package_name": kernel_name}
    )

    # Print the description
    description = _extract_description_before_tags(docstring_mdx)
    print(f"\n{description}")

    # Print parameters and returns sections
    _print_parameters_section(docstring_mdx, header_level=header_level)
    _print_returns_section(
        docstring_mdx, context_name=context_name, header_level=header_level
    )


def generate_readme_for_kernel(repo_id: str, *, revision: str = "main") -> None:
    kernel_module = get_kernel(repo_id=repo_id, revision=revision)
    kernel_name = repo_id.split("/")[-1].replace("-", "_")

    generate_metadata(kernel_module)
    generate_kernel_doc(kernel_module, kernel_name)
    generate_function_doc(kernel_module, kernel_name)
    generate_layers_doc(kernel_module, kernel_name)


def generate_metadata(module: ModuleType) -> None:
    metadata = getattr(module, "__kernel_metadata__", {})
    if "tags" not in metadata:
        metadata["tags"] = ["kernel"]
    else:
        if "kernel" not in metadata["tags"]:
            metadata["tags"].append("kernel")

    print("---")
    print(yaml.dump(metadata), end="")
    print("---")


def generate_kernel_doc(module: ModuleType, kernel_name: str) -> None:
    docstring = module.__doc__.strip() if module.__doc__ is not None else None
    if docstring:
        title, rest = docstring.split("\n", 1)
        print(f"# {title.strip()}")
        print(
            f"\n{convert_rst_docstring_to_mdx(rest.strip(), page_info={'package_name': kernel_name})}"
        )


def generate_function_doc(kernel_module: ModuleType, kernel_name: str) -> None:
    print("\n## Functions")

    # Track if we found any functions
    found_functions = False

    for name, func in inspect.getmembers(kernel_module, inspect.isfunction):
        # Do not include imported functions.
        if func.__module__ != kernel_module.__name__:
            continue

        # Exclude private functions.
        if name.startswith("_"):
            continue

        found_functions = True

        try:
            sig = inspect.signature(func)
            docstring = _get_docstring(func)
        except ValueError:
            print(
                f"Warning: Could not retrieve signature for {name} in {kernel_module.__name__}",
                file=sys.stderr,
            )
            continue

        print(f"\n### Function `{name}`")
        print(f"\n`{sig}`")

        _process_and_print_docstring(
            docstring, kernel_name=kernel_name, context_name=name, header_level=3
        )

    if not found_functions:
        print("\nNo public top-level functions.")


def generate_layers_doc(kernel_module: ModuleType, kernel_name: str) -> None:
    # Check if layers module is available
    layers_module = getattr(kernel_module, "layers", None)
    if layers_module is None:
        return

    print("\n## Layers")

    # Track if we found any classes
    found_classes = False

    for class_name, cls in inspect.getmembers(layers_module, inspect.isclass):
        # Exclude classes that were imported.
        if cls.__module__ != layers_module.__name__:
            continue

        found_classes = True

        try:
            # Get docstring, but not from superclasses.
            class_docstring = _get_docstring(cls, use_dict_check=True)
        except Exception:
            print(
                f"Warning: Could not retrieve documentation for class {class_name} in {layers_module.__name__}",
                file=sys.stderr,
            )
            continue

        print(f"\n### Class `{class_name}`")

        # Always print class description (helper handles conversion and formatting)
        class_docstring_mdx = convert_rst_docstring_to_mdx(
            class_docstring, page_info={"package_name": kernel_name}
        )
        description = _extract_description_before_tags(class_docstring_mdx)
        print(f"\n{description}")

        # Document methods
        print("\n#### Methods")

        for method_name, method in inspect.getmembers(cls, inspect.isfunction):
            # Note: also skip __init__, since extension layers cannot have a constructor.
            if method_name.startswith("_"):
                continue

            # Skip methods from superclasses.
            if method_name not in cls.__dict__:
                continue

            try:
                sig = inspect.signature(method)
                method_docstring = _get_docstring(method)
            except ValueError:
                print(
                    f"Warning: Could not retrieve signature for {method_name} in {class_name}",
                    file=sys.stderr,
                )
                continue

            print(f"\n##### Method `{method_name}`")
            print(f"\n`{sig}`")

            _process_and_print_docstring(
                method_docstring,
                kernel_name=kernel_name,
                context_name=method_name,
                header_level=6,
            )

    if not found_classes:
        print("\nNo layers defined.")
