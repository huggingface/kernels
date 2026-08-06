"""Shared helpers for the AST-based source rewriters in `tools/`.

The rewriters parse Python with :mod:`ast` for *analysis* only. Rewriting is
done as surgical replacements of source spans, so comments, string quoting
style, formatting, and unrelated code survive byte-for-byte. Round-tripping
through :func:`ast.unparse` would destroy all of that, so it is never used to
emit whole files.

Only the standard library is used, so every tool here runs with a bare
`python tools/<tool>.py` and no install step.
"""

from __future__ import annotations

import ast
import difflib
import io
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Edit:
    """A replacement of ``text[start:end]`` by ``new_text``."""

    start: int
    end: int
    new_text: str


def apply_edits(text: str, edits: Iterable[Edit]) -> str:
    """Apply non-overlapping `edits` to `text`.

    Raises:
        ValueError: if two edits overlap. Overlapping edits mean the caller
            derived two conflicting rewrites for the same source span, which
            is always a bug rather than something to paper over.
    """
    ordered = sorted(edits, key=lambda edit: (edit.start, edit.end))
    parts: List[str] = []
    pos = 0
    for edit in ordered:
        if edit.start < pos:
            raise ValueError(f"overlapping edits at offset {edit.start}")
        parts.append(text[pos : edit.start])
        parts.append(edit.new_text)
        pos = edit.end
    parts.append(text[pos:])
    return "".join(parts)


class SourceFile:
    """A parsed Python source file with offset-accurate span lookups."""

    def __init__(self, path: Path, text: str):
        self.path = path
        self.text = text
        self.tree = ast.parse(text, filename=str(path))

        self._line_starts = [0]
        for line in text.splitlines(keepends=True):
            self._line_starts.append(self._line_starts[-1] + len(line))

        self._tokens: Optional[List[Tuple[int, str, int, int]]] = None

    @classmethod
    def read(cls, path: Path) -> "SourceFile":
        return cls(path, path.read_text(encoding="utf-8"))

    def offset(self, lineno: int, col_offset: int) -> int:
        """Convert an AST ``(lineno, col_offset)`` pair to a string offset.

        `ast` reports columns as UTF-8 *byte* offsets, so a line with
        non-ASCII characters needs a re-decode to land on the right
        character index.
        """
        line_start = self._line_starts[lineno - 1]
        line_end = (
            self._line_starts[lineno]
            if lineno < len(self._line_starts)
            else len(self.text)
        )
        line = self.text[line_start:line_end]
        prefix = line.encode("utf-8")[:col_offset].decode("utf-8", errors="ignore")
        return line_start + len(prefix)

    def span(self, node: ast.AST) -> Tuple[int, int]:
        """Return the ``(start, end)`` string offsets of `node`."""
        start = self.offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end = self.offset(node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
        return start, end

    def segment(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.text[start:end]

    def line_of(self, offset: int) -> int:
        """Return the 1-based line number containing `offset`."""
        lo, hi = 0, len(self._line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def indent_of(self, node: ast.stmt) -> Optional[str]:
        """Return the indentation of `node`, or `None` if it does not start its line.

        A statement that shares a line with other code (``if x: import y``)
        cannot be safely expanded into several statements, so callers use
        `None` as a signal to report the case instead of rewriting it.
        """
        start = self.offset(node.lineno, node.col_offset)
        line_start = self._line_starts[node.lineno - 1]
        prefix = self.text[line_start:start]
        return prefix if prefix.strip() == "" else None

    def _char_offset(self, lineno: int, col: int) -> int:
        """Convert a `tokenize` ``(lineno, col)`` pair to a string offset.

        Unlike `ast`, `tokenize` reports character columns, so no re-decode
        is needed.
        """
        return self._line_starts[lineno - 1] + col

    @property
    def tokens(self) -> List[Tuple[int, str, int, int]]:
        """Tokens as ``(type, string, start_offset, end_offset)`` tuples."""
        if self._tokens is None:
            self._tokens = [
                (
                    tok.type,
                    tok.string,
                    self._char_offset(*tok.start),
                    self._char_offset(*tok.end),
                )
                for tok in tokenize.generate_tokens(io.StringIO(self.text).readline)
            ]
        return self._tokens

    def module_span(self, node: ast.ImportFrom) -> Tuple[int, int]:
        """Return the offsets of the module part of a `from ... import ...`.

        The span covers everything between the `from` and `import` keywords,
        including the surrounding whitespace, so replacing it leaves the
        imported-name list (and any comments in it) untouched.
        """
        start, end = self.span(node)
        from_end = None
        for tok_type, tok_str, tok_start, tok_end in self.tokens:
            if tok_start < start or tok_start >= end:
                continue
            if tok_type != tokenize.NAME:
                continue
            if from_end is None:
                if tok_str != "from":
                    break
                from_end = tok_end
                continue
            if tok_str == "import":
                return from_end, tok_start
        raise ValueError(
            f"{self.path}: could not locate `import` keyword on line {node.lineno}"
        )


def dotted_name(node: ast.AST) -> Optional[str]:
    """Return the dotted path of a pure ``Name``/``Attribute`` chain."""
    parts: List[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def iter_python_files(paths: Sequence[Path], exclude: Sequence[str] = ()) -> List[Path]:
    """Collect `.py` files from `paths`, recursing into directories."""
    found: List[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            found.append(path)
    result = []
    for path in found:
        if any(path.match(pattern) for pattern in exclude):
            continue
        result.append(path)
    return result


def unified_diff(path: Path, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def relative_import(from_parts: Sequence[str], target_parts: Sequence[str]) -> str:
    """Build the shortest relative module reference between two package paths.

    `from_parts` is the package path of the *directory* holding the importing
    file, relative to the package root; `target_parts` is the module path of
    the import target, also relative to the package root. Both are empty at
    the root.

    A module in ``<root>/a/b/`` reaching ``<root>/a/b/c`` gets ``.c``, and
    reaching the root itself gets ``...`` (one dot for the current package,
    plus one per level climbed).
    """
    common = 0
    for left, right in zip(from_parts, target_parts):
        if left != right:
            break
        common += 1
    dots = "." * (1 + len(from_parts) - common)
    return dots + ".".join(target_parts[common:])
