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


def generate_readme_for_kernel(repo_id: str, *, revision: str = "main") -> None:
    kernel_module = get_kernel(repo_id=repo_id, revision=revision)
    kernel_name = repo_id.split("/")[-1].replace("-", "_")

    generate_metadata(kernel_module)
    generate_kernel_doc(kernel_module, kernel_name)
    generate_function_doc(kernel_module, kernel_name)


def generate_metadata(module: ModuleType):
    metadata = getattr(module, "__kernel_metadata__", {})
    if "tags" not in metadata:
        metadata["tags"] = ["kernel"]
    else:
        if "kernel" not in metadata["tags"]:
            metadata["tags"].append("kernel")

    print("---")
    print(yaml.dump(metadata), end="")
    print("---")


def generate_kernel_doc(module: ModuleType, kernel_name: str):
    docstring = module.__doc__.strip() if module.__doc__ is not None else None
    if docstring:
        title, rest = docstring.split("\n", 1)
        print(f"# {title.strip()}")
        print(
            f"\n{convert_rst_docstring_to_mdx(rest.strip(), page_info={'package_name': kernel_name})}"
        )


def generate_function_doc(kernel_module, kernel_name):
    functions_info = []
    for name, func in inspect.getmembers(kernel_module, inspect.isfunction):
        # Do not include imported functions.
        if func.__module__ == kernel_module.__name__:
            # Exclude private functions.
            if not name.startswith("_"):
                try:
                    sig = inspect.signature(func)
                    docstring = inspect.getdoc(func) or "No documentation available."
                    functions_info.append((name, sig, docstring))
                except ValueError:
                    print(
                        f"Warning: Could not retrieve signature for {name} in {kernel_module.__name__}",
                        file=sys.stderr,
                    )

    print("\n## Functions")

    if not functions_info:
        print(
            "\nNo public top-level functions.",
        )
        return

    for name, sig, docstring in functions_info:
        print(f"\n### Function `{name}`")
        print(f"\n`{sig}`")

        docstring_mdx = convert_rst_docstring_to_mdx(
            docstring, page_info={"package_name": kernel_name}
        )

        params_pos = docstring_mdx.find("<parameters>")
        returns_pos = docstring_mdx.find("<returns>")
        returntype_pos = docstring_mdx.find("<returntype>")
        positions = [
            pos for pos in [params_pos, returns_pos, returntype_pos] if pos != -1
        ]

        if positions:
            first_tag_pos = min(positions)
            # The function description is anything before the first tag.
            print(f"\n{docstring_mdx[:first_tag_pos].strip()}")
        else:
            print(f"\n{docstring_mdx.strip()}")

        # Extract parameters
        matches = _RE_PARAMETERS.findall(docstring_mdx)
        if matches:
            print("\n### Parameters")
            for match in matches:
                print(f"\n{match[0].strip()}")

        # Extract return information
        return_matches = _RE_RETURNS.findall(docstring_mdx)
        returntype_matches = _RE_RETURNTYPE.findall(docstring_mdx)

        if return_matches or returntype_matches:
            print("\n### Returns", file=sys.stdout)

            if returntype_matches:
                if len(returntype_matches) > 1:
                    raise ValueError(
                        f"More than one <returntype> tag found in docstring for {name} in {kernel_module.__name__}"
                    )
                print(
                    f"\n**Type**: {returntype_matches[0][0].strip()}", file=sys.stdout
                )

            if return_matches:
                for match in return_matches:
                    print(f"\n{match[0].strip()}")
