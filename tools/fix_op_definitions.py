#!/usr/bin/env python3
"""Fix Torch op registrations to use the build-generated namespace prefix.

Several versions of the same kernel can be loaded into one Python process,
so every kernel gets a unique Torch op namespace at build time. Op names
must never hardcode a namespace: they go through `add_op_namespace_prefix`
from the generated `_ops` module. See
`docs/source/builder/writing-kernels.md`.

This tool finds `torch.library` registrations with `ast`, rewrites the op
name argument, and adds the `_ops` import when it is missing:

    @torch.library.custom_op("relu::relu_fwd", mutates_args=())
    ->
    @torch.library.custom_op(add_op_namespace_prefix("relu_fwd"), mutates_args=())

An existing hardcoded namespace (`relu::`) is stripped, since
`add_op_namespace_prefix` supplies the real one.

Cases that cannot be fixed safely are reported rather than rewritten: op
names that are not string literals, `torch.library.Library(...)` (whose
namespace is fixed at construction), and the re-wrapping antipattern that
`writing-kernels.md` warns about.

Nothing is written without `--write`.

Examples:
    python tools/fix_op_definitions.py torch-ext/hello
    python tools/fix_op_definitions.py torch-ext/hello --write --json
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from _ast_utils import (
    Edit,
    SourceFile,
    apply_edits,
    dotted_name,
    iter_python_files,
    relative_import,
    unified_diff,
)
from _report import Change, Issue, Report, display_path, emit, fail

TOOL = "fix-op-definitions"

HELPER = "add_op_namespace_prefix"
OPS_MODULE = "_ops"

# `torch.library` functions whose first positional argument is an op name.
#
# Matching on the bare function name is not enough: most of these also exist
# as methods, on the `CustomOpDef` returned by `custom_op`
# (`my_op.register_autograd(backward, ...)`) and on `torch.library.Library`
# (`lib.define("myop() -> ()")`). In those forms the first argument is not an
# op name and the namespace comes from the receiver, so a call only counts
# when it is reached through the `torch.library` module itself.
OP_NAME_FUNCTIONS = frozenset(
    {
        "custom_op",
        "triton_op",
        "define",
        "impl",
        "impl_abstract",
        "register_fake",
        "register_kernel",
        "register_autograd",
        "register_vmap",
        "register_torch_dispatch",
        "register_autocast",
    }
)


def _func_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class FileFixer:
    """Collects the op-registration edits for a single file."""

    def __init__(self, source: SourceFile, ops_ref: str, label: str):
        self.source = source
        self.ops_ref = ops_ref
        self.label = label
        self.edits: List[Edit] = []
        self.changes: List[Change] = []
        self.issues: List[Issue] = []
        # Local name -> `torch.library` function it was imported as.
        self.torch_library_funcs: Dict[str, str] = {}
        # Local names that refer to the `torch.library` module itself.
        self.torch_library_modules: Set[str] = {"torch.library"}
        self.helper_bound = False

    def run(self) -> None:
        self._scan_imports()
        self._check_antipatterns()
        for node in ast.walk(self.source.tree):
            if isinstance(node, ast.Call):
                self._visit_call(node)
        if self.edits and not self.helper_bound:
            self._insert_helper_import()

    def _issue(self, node: ast.AST, kind: str, message: str) -> None:
        self.issues.append(
            Issue(
                file=self.label,
                line=getattr(node, "lineno", 0),
                kind=kind,
                message=message,
                snippet=" ".join(self.source.segment(node).split())[:200],
            )
        )

    def _scan_imports(self) -> None:
        """Record how `torch.library` and the `_ops` helper are bound locally."""
        for node in ast.walk(self.source.tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module == "torch.library":
                    for alias in node.names:
                        self.torch_library_funcs[alias.asname or alias.name] = (
                            alias.name
                        )
                if node.level == 0 and node.module == "torch":
                    for alias in node.names:
                        if alias.name == "library":
                            self.torch_library_modules.add(alias.asname or alias.name)
                for alias in node.names:
                    if (alias.asname or alias.name) == HELPER:
                        self.helper_bound = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "torch.library" and alias.asname:
                        self.torch_library_modules.add(alias.asname)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == HELPER:
                    self.helper_bound = True

    def _check_antipatterns(self) -> None:
        for node in ast.walk(self.source.tree):
            if isinstance(node, ast.Try):
                imports_helper = any(
                    isinstance(child, ast.ImportFrom)
                    and any(
                        (alias.asname or alias.name) == HELPER for alias in child.names
                    )
                    for child in ast.walk(node)
                )
                if imports_helper and node.handlers:
                    self._issue(
                        node,
                        "fallback-import",
                        f"`{HELPER}` is imported with a fallback; a fallback masks a broken "
                        "import path and yields non-unique op names. Import it directly from "
                        f"`{OPS_MODULE}`.",
                    )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == HELPER:
                    self._issue(
                        node,
                        "rewrapped-helper",
                        f"`{HELPER}` is redefined here; it must be used directly from "
                        f"`{OPS_MODULE}` so that ops can be analyzed statically.",
                    )
            elif isinstance(node, ast.Call):
                path = dotted_name(node.func)
                is_torch_library = path is not None and (
                    path.rpartition(".")[0] in self.torch_library_modules
                    and path.rpartition(".")[2] == "Library"
                    or self.torch_library_funcs.get(path) == "Library"
                )
                if is_torch_library:
                    self._issue(
                        node,
                        "hardcoded-library-namespace",
                        "`torch.library.Library(...)` fixes the op namespace at construction; "
                        f"pass `{HELPER}(...)`-derived names or register ops with "
                        "`torch.library.custom_op` instead.",
                    )

    def _is_op_registration(self, node: ast.Call) -> bool:
        """Whether `node` is a `torch.library` call taking an op name first."""
        if isinstance(node.func, ast.Name):
            original = self.torch_library_funcs.get(node.func.id)
            return original is not None and original in OP_NAME_FUNCTIONS
        if isinstance(node.func, ast.Attribute):
            if node.func.attr not in OP_NAME_FUNCTIONS:
                return False
            receiver = dotted_name(node.func.value)
            return receiver in self.torch_library_modules
        return False

    def _visit_call(self, node: ast.Call) -> None:
        if not self._is_op_registration(node):
            return
        if not node.args:
            return
        arg = node.args[0]

        if isinstance(arg, ast.Call) and _func_name(arg) == HELPER:
            return  # Already prefixed.

        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            self._issue(
                node,
                "non-literal-op-name",
                f"`{_func_name(node)}` is called with a non-literal op name, so the namespace "
                f"prefix cannot be added automatically. Wrap it in `{HELPER}(...)` by hand.",
            )
            return

        op_name = arg.value
        # A hardcoded namespace is replaced, not kept: `add_op_namespace_prefix`
        # supplies the unique one that the build generates.
        _, _, bare = op_name.rpartition("::")
        quote = "'" if '"' in bare else '"'
        replacement = f"{HELPER}({quote}{bare}{quote})"

        start, end = self.source.span(arg)
        self.edits.append(Edit(start, end, replacement))
        self.changes.append(
            Change(
                file=self.label,
                line=node.lineno,
                kind="op-name",
                before=self.source.text[start:end],
                after=replacement,
            )
        )

    def _insert_helper_import(self) -> None:
        """Add `from <dots>_ops import add_op_namespace_prefix` after the imports."""
        statement = f"from {self.ops_ref} import {HELPER}"
        body = self.source.tree.body
        anchor = None
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                anchor = node
            elif (
                anchor is None
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                anchor = node  # Module docstring.

        if anchor is None:
            offset, text = 0, f"{statement}\n"
        else:
            _, offset = self.source.span(anchor)
            text = f"\n{statement}"

        self.edits.append(Edit(offset, offset, text))
        self.changes.append(
            Change(
                file=self.label,
                line=self.source.line_of(offset),
                kind="import",
                before="",
                after=statement,
            )
        )
        self.helper_bound = True


def find_package_root(path: Path) -> Path:
    """Return the outermost directory of the package `path` belongs to."""
    directory = path if path.is_dir() else path.parent
    root = directory
    current = directory
    while (current / "__init__.py").is_file():
        root = current
        if current.parent == current:
            break
        current = current.parent
    return root


def process(
    paths: Sequence[Path],
    *,
    package_root: Optional[Path],
    write: bool,
    exclude: Sequence[str],
) -> Report:
    report = Report(tool=TOOL, mode="write" if write else "check")
    for path in iter_python_files(paths, exclude=exclude):
        if path.name == f"{OPS_MODULE}.py":
            continue  # The generated module that defines the helper.

        label = display_path(path)
        try:
            source = SourceFile.read(path)
        except SyntaxError as err:
            report.issues.append(
                Issue(
                    file=label,
                    line=err.lineno or 0,
                    kind="syntax-error",
                    message=f"could not parse: {err.msg}",
                )
            )
            report.files_scanned += 1
            continue

        report.files_scanned += 1
        root = package_root or find_package_root(path)
        try:
            dir_parts = path.parent.resolve().relative_to(root.resolve()).parts
        except ValueError:
            dir_parts = ()
        ops_ref = relative_import(dir_parts, (OPS_MODULE,))

        fixer = FileFixer(source, ops_ref, label)
        fixer.run()
        report.issues.extend(fixer.issues)
        if not fixer.edits:
            continue

        new_text = apply_edits(source.text, fixer.edits)
        report.changes.extend(fixer.changes)
        report.diffs[label] = unified_diff(Path(label), source.text, new_text)
        if write:
            path.write_text(new_text, encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/fix_op_definitions.py",
        description="Rewrite Torch op registrations to use add_op_namespace_prefix.",
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="Files or directories to process, e.g. torch-ext/<kernel_name>.",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        help=(
            "Package root that holds the generated _ops.py (default: detected from the enclosing __init__.py files)."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="Skip files matching this glob. Repeatable.",
    )
    parser.add_argument("--write", action="store_true", help="Apply the rewrites.")
    parser.add_argument("--diff", action="store_true", help="Show a unified diff.")
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    args = parser.parse_args(argv)

    for path in args.paths:
        if not path.exists():
            return fail(f"{path} does not exist", as_json=args.json, tool=TOOL)

    try:
        report = process(
            args.paths,
            package_root=args.package_root,
            write=args.write,
            exclude=args.exclude,
        )
    except (OSError, ValueError) as err:
        return fail(str(err), as_json=args.json, tool=TOOL)

    return emit(report, as_json=args.json, show_diff=args.diff)


if __name__ == "__main__":
    sys.exit(main())
