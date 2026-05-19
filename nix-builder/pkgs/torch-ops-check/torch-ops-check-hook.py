import argparse
import ast
import sys
from pathlib import Path

TORCH_LIBRARY_DECORATORS = frozenset(
    {
        "custom_op",
        "triton_op",
        "register_kernel",
        "register_fake",
        "register_vmap",
        "register_torch_dispatch",
    }
)


def get_func_name(node: ast.Call) -> str | None:
    """Return the the function name of a call."""
    # myfunc(...)
    if isinstance(node.func, ast.Name):
        return node.func.id
    # somelib.myfunc(...)
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def is_torch_library_decorator(node: ast.expr) -> bool:
    """Check if a decorator call is one of the known torch.library decorators."""
    return (
        isinstance(node, ast.Call)
        and len(node.args) > 0
        and get_func_name(node) in TORCH_LIBRARY_DECORATORS
    )


def is_add_namespace_prefix(node: ast.expr) -> bool:
    """Check if a node is a call to add_op_namespace_prefix (bare or attribute)."""
    return (
        isinstance(node, ast.Call) and get_func_name(node) == "add_op_namespace_prefix"
    )


NODE_TYPES: dict[type, str] = {
    ast.FunctionDef: "function",
    ast.AsyncFunctionDef: "async function",
    ast.ClassDef: "class",
}


def find_violations(filename: str, tree: ast.Module) -> list[str]:
    violations = []
    for node in ast.iter_child_nodes(tree):
        if type(node) in NODE_TYPES:
            for decorator in node.decorator_list:
                if is_torch_library_decorator(
                    decorator
                ) and not is_add_namespace_prefix(decorator.args[0]):
                    node_type = NODE_TYPES[type(node)]
                    violations.append(
                        f"{filename}:{node.lineno} {node_type} {node.name}"
                    )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that Torch library decorators use add_op_namespace_prefix."
    )
    parser.add_argument("directory", help="Directory to recursively check")
    args = parser.parse_args()

    violations = []
    for path in sorted(Path(args.directory).rglob("*.py")):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        violations.extend(find_violations(str(path), tree))

    if violations:
        print(
            "Found Torch library registrations that do not use `add_op_namespace_prefix`:"
        )
        for v in violations:
            print(v)
        sys.exit(1)


if __name__ == "__main__":
    main()
