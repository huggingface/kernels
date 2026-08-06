#!/usr/bin/env python3
"""Rewrite absolute intra-package imports to relative ones.

Hub kernels are loaded from a directory whose name is *not* the package
name (the build variant directory), so any absolute import of the kernel's
own modules breaks at load time. Every intra-kernel import in
`torch-ext/<kernel_name>` must therefore be relative -- see
`docs/source/kernel-requirements.md`.

This tool parses each file with `ast` to find the imports, then edits only
the affected source spans, so comments and formatting are preserved.

What gets rewritten (for a file `<root>/a/b/mod.py`):

    from <pkg>.utils import x      ->  from ...utils import x
    from <pkg> import x            ->  from ... import x
    import <pkg>.utils.helpers     ->  from ...utils import helpers
                                       (plus `<pkg>.utils.helpers.` references
                                        renamed to `helpers.`)
    import <pkg>.utils as u        ->  from ... import utils as u

Which absolute names count as intra-package is decided by:

* auto-detection (default): a top-level name that is the package directory
  name, or that exists as a module or subpackage inside it;
* `--module-map SRC=DST`, for vendored sources that were moved during the
  copy, e.g. `--module-map liger_kernel.ops=` maps `liger_kernel.ops.*`
  onto the package root.

Nothing is written without `--write`.

Examples:
    python tools/relativize_imports.py torch-ext/flash_attn4
    python tools/relativize_imports.py torch-ext/liger_kernels \\
        --module-map liger_kernel.ops= --module-map liger_kernel=. --write
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

TOOL = "relativize-imports"

Parts = Tuple[str, ...]


class ModuleResolver:
    """Decides which absolute module names live inside the kernel package."""

    def __init__(
        self,
        root: Path,
        *,
        package_name: str,
        module_map: Dict[Parts, Parts],
        auto: bool = True,
    ):
        self.root = root
        self.package_name = package_name
        self.module_map = module_map
        self.auto = auto

    def resolve(self, module: str) -> Optional[Parts]:
        """Map an absolute module name to a path relative to the package root.

        Returns `None` for modules that are external to the package (`torch`,
        `triton`, ...), and an empty tuple for the package root itself.
        """
        parts = tuple(module.split("."))

        # Explicit maps win, longest prefix first, so a caller can map
        # `liger_kernel.ops` to the root while mapping `liger_kernel`
        # somewhere else.
        for length in range(len(parts), 0, -1):
            prefix = parts[:length]
            if prefix in self.module_map:
                return self.module_map[prefix] + parts[length:]

        if not self.auto:
            return None
        if parts[0] == self.package_name:
            return parts[1:]
        if self.contains(parts[:1]):
            return parts
        return None

    def contains(self, parts: Parts) -> bool:
        """Whether `parts` names a module or package inside the root."""
        if not parts:
            return True
        path = self.root.joinpath(*parts)
        return path.with_suffix(".py").is_file() or path.is_dir()


def parse_module_map(entries: Sequence[str]) -> Dict[Parts, Parts]:
    """Parse `--module-map SRC=DST` entries.

    `DST` is a dotted path relative to the package root; an empty `DST` (or
    `.`) maps onto the root itself.
    """
    mapping: Dict[Parts, Parts] = {}
    for entry in entries:
        source, sep, target = entry.partition("=")
        if not sep or not source:
            raise ValueError(f"invalid --module-map entry {entry!r}, expected SRC=DST")
        target = target.strip().strip(".")
        mapping[tuple(source.split("."))] = tuple(target.split(".")) if target else ()
    return mapping


def _summarize(text: str) -> str:
    """Collapse a (possibly multi-line) statement into one readable line."""
    return " ".join(text.split())


class FileRewriter:
    """Collects the edits for a single file."""

    def __init__(
        self, source: SourceFile, resolver: ModuleResolver, dir_parts: Parts, label: str
    ):
        self.source = source
        self.resolver = resolver
        self.dir_parts = dir_parts
        self.label = label
        self.edits: List[Edit] = []
        self.changes: List[Change] = []
        self.issues: List[Issue] = []
        # Dotted module path -> the simple name it is bound to after rewriting.
        self.renames: Dict[str, str] = {}
        # Names an `import` used to bind that the relative form no longer does.
        self.dropped_bindings: set = set()
        # Names that are still bound after the rewrite, by any import.
        self.live_bindings: set = set()

    def run(self) -> None:
        for node in ast.walk(self.source.tree):
            if isinstance(node, ast.ImportFrom):
                # `from x import y` keeps binding `y` whether or not the
                # module part is rewritten.
                self.live_bindings.update(
                    alias.asname or alias.name for alias in node.names
                )
                self._rewrite_import_from(node)
            elif isinstance(node, ast.Import):
                self._rewrite_import(node)
        self._apply_renames()

    def _issue(self, node: ast.AST, kind: str, message: str) -> None:
        self.issues.append(
            Issue(
                file=self.label,
                line=getattr(node, "lineno", 0),
                kind=kind,
                message=message,
                snippet=_summarize(self.source.segment(node)),
            )
        )

    def _check_target(self, node: ast.AST, target: Parts, module: str) -> None:
        if not self.resolver.contains(target):
            self._issue(
                node,
                "unresolved-target",
                f"`{module}` maps to `{'.'.join(target) or '<package root>'}`, which does not exist in the package",
            )

    def _rewrite_import_from(self, node: ast.ImportFrom) -> None:
        if node.level > 0 or node.module is None:
            return  # Already relative.
        target = self.resolver.resolve(node.module)
        if target is None:
            return
        self._check_target(node, target, node.module)

        new_ref = relative_import(self.dir_parts, target)
        start, end = self.source.module_span(node)
        self.edits.append(Edit(start, end, f" {new_ref} "))
        names = ", ".join(
            alias.name + (f" as {alias.asname}" if alias.asname else "")
            for alias in node.names
        )
        self.changes.append(
            Change(
                file=self.label,
                line=node.lineno,
                kind="import-from",
                before=f"from {node.module} import {names}",
                after=f"from {new_ref} import {names}",
            )
        )

    def _rewrite_import(self, node: ast.Import) -> None:
        targets = [self.resolver.resolve(alias.name) for alias in node.names]
        if all(target is None for target in targets):
            return

        statements: List[str] = []
        renames: Dict[str, str] = {}
        dropped: set = set()
        for alias, target in zip(node.names, targets):
            original = alias.name + (f" as {alias.asname}" if alias.asname else "")
            binding = alias.asname or alias.name.split(".")[0]
            if target is None:
                statements.append(f"import {original}")
                self.live_bindings.add(binding)
                continue
            self._check_target(node, target, alias.name)
            if not target:
                # `import <pkg>` binds the package root, which has no
                # relative spelling -- there is no `from <dots> import`
                # form that yields the current package object.
                self._issue(
                    node,
                    "unrepresentable-import",
                    f"`import {original}` refers to the package root; rewrite it by hand "
                    "(import the submodules that are actually used instead)",
                )
                statements.append(f"import {original}")
                self.live_bindings.add(binding)
                continue

            ref = relative_import(self.dir_parts, target[:-1])
            leaf = target[-1]
            if alias.asname:
                statements.append(f"from {ref} import {leaf} as {alias.asname}")
                self.live_bindings.add(alias.asname)
            else:
                statements.append(f"from {ref} import {leaf}")
                self.live_bindings.add(leaf)
                if alias.name != leaf:
                    # `import a.b.c` binds `a`; the relative form binds `c`,
                    # so every `a.b.c` reference has to follow.
                    renames[alias.name] = leaf
                    dropped.add(binding)

        if len(statements) > 1:
            indent = self.source.indent_of(node)
            if indent is None:
                self._issue(
                    node,
                    "compound-import",
                    "cannot split a multi-name `import` that does not start its own line",
                )
                return
            replacement = ("\n" + indent).join(statements)
        else:
            replacement = statements[0]

        start, end = self.source.span(node)
        original_text = _summarize(self.source.text[start:end])
        if replacement == original_text:
            return
        self.edits.append(Edit(start, end, replacement))
        self.renames.update(renames)
        self.dropped_bindings.update(dropped)
        self.changes.append(
            Change(
                file=self.label,
                line=node.lineno,
                kind="import",
                before=original_text,
                after=_summarize(replacement),
            )
        )

    def _apply_renames(self) -> None:
        """Rewrite `a.b.c` references left dangling by a rewritten `import a.b.c`."""
        if not self.renames:
            return

        matches: List[Tuple[int, int, str]] = []
        for node in ast.walk(self.source.tree):
            if not isinstance(node, (ast.Attribute, ast.Name)):
                continue
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                continue
            path = dotted_name(node)
            if path is None or path not in self.renames:
                continue
            start, end = self.source.span(node)
            matches.append((start, end, self.renames[path]))

        # Keep only the outermost match of each chain, so nested imports
        # (`a.b` and `a.b.c` both rewritten) do not produce overlapping edits.
        matches.sort(key=lambda match: (match[0], -match[1]))
        kept: List[Tuple[int, int, str]] = []
        for start, end, name in matches:
            if kept and start < kept[-1][1]:
                continue
            kept.append((start, end, name))

        covered = [(start, end) for start, end, _ in kept]
        for start, end, name in kept:
            self.edits.append(Edit(start, end, name))
            self.changes.append(
                Change(
                    file=self.label,
                    line=self.source.line_of(start),
                    kind="reference",
                    before=self.source.text[start:end],
                    after=name,
                )
            )

        self._report_dangling(covered)

    def _report_dangling(self, covered: Sequence[Tuple[int, int]]) -> None:
        """Flag uses of a root binding that the rewrite left unbound.

        Rewriting `import a.b.c` to `from .. import c` drops the `a`
        binding. References of the form `a.b.c...` were renamed above; any
        *other* use of `a` (e.g. `a.d.f()` from a second import that was not
        rewritten) would break, so it is reported instead.
        """
        unbound = self.dropped_bindings - self.live_bindings
        if not unbound:
            return
        for node in ast.walk(self.source.tree):
            if not isinstance(node, ast.Name) or node.id not in unbound:
                continue
            start, end = self.source.span(node)
            if any(low <= start and end <= high for low, high in covered):
                continue
            self._issue(
                node,
                "dangling-reference",
                f"`{node.id}` is no longer bound after relativizing "
                f"`import {node.id}...`; rewrite this reference by hand",
            )


def process(
    root: Path,
    resolver: ModuleResolver,
    *,
    write: bool,
    exclude: Sequence[str],
) -> Report:
    report = Report(tool=TOOL, mode="write" if write else "check")
    for path in iter_python_files([root], exclude=exclude):
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
        dir_parts = path.parent.resolve().relative_to(root.resolve()).parts
        dir_parts = tuple(part for part in dir_parts if part != ".")

        rewriter = FileRewriter(source, resolver, dir_parts, label)
        rewriter.run()
        report.issues.extend(rewriter.issues)
        if not rewriter.edits:
            continue

        new_text = apply_edits(source.text, rewriter.edits)
        report.changes.extend(rewriter.changes)
        report.diffs[label] = unified_diff(Path(label), source.text, new_text)
        if write:
            path.write_text(new_text, encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/relativize_imports.py",
        description="Rewrite absolute intra-package imports to relative imports.",
    )
    parser.add_argument(
        "package_dir",
        type=Path,
        help="Package root, e.g. torch-ext/<kernel_name>.",
    )
    parser.add_argument(
        "--package-name",
        help="Name the sources import the package as (default: the directory name).",
    )
    parser.add_argument(
        "--module-map",
        action="append",
        default=[],
        metavar="SRC=DST",
        help=(
            "Map absolute module SRC (and its submodules) onto the package-relative "
            "path DST. An empty DST means the package root. Repeatable."
        ),
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Only relativize modules listed with --module-map.",
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

    root = args.package_dir
    if not root.is_dir():
        return fail(f"{root} is not a directory", as_json=args.json, tool=TOOL)

    try:
        module_map = parse_module_map(args.module_map)
    except ValueError as err:
        return fail(str(err), as_json=args.json, tool=TOOL)

    resolver = ModuleResolver(
        root,
        package_name=args.package_name or root.resolve().name,
        module_map=module_map,
        auto=not args.no_auto,
    )

    try:
        report = process(root, resolver, write=args.write, exclude=args.exclude)
    except (OSError, ValueError) as err:
        return fail(str(err), as_json=args.json, tool=TOOL)

    return emit(report, as_json=args.json, show_diff=args.diff)


if __name__ == "__main__":
    sys.exit(main())
