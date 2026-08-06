"""Shared CLI reporting for the `tools/` rewriters.

Every rewriter speaks the same protocol so an agent only has to learn it
once:

* Running without `--write` is a dry run: nothing is touched and the exit
  code says whether the tree is already clean.
* `--json` prints a single JSON object on stdout, and nothing else. Human
  progress output goes to stderr, so `--json` output is always safe to pipe.
* Exit codes: `0` clean (or everything applied), `1` work is pending or
  something needs a human, `2` the tool could not run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

EXIT_CLEAN = 0
EXIT_PENDING = 1
EXIT_ERROR = 2


@dataclass
class Change:
    """A rewrite the tool made (or would make in a dry run)."""

    file: str
    line: int
    kind: str
    before: str
    after: str


@dataclass
class Issue:
    """Something the tool refuses to rewrite and a human must look at."""

    file: str
    line: int
    kind: str
    message: str
    snippet: str = ""


@dataclass
class Report:
    tool: str
    mode: str
    files_scanned: int = 0
    changes: List[Change] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    diffs: dict = field(default_factory=dict)

    @property
    def changed_files(self) -> List[str]:
        seen = []
        for change in self.changes:
            if change.file not in seen:
                seen.append(change.file)
        return seen

    @property
    def ok(self) -> bool:
        """True when there is nothing left for the caller to do."""
        if self.issues:
            return False
        return self.mode == "write" or not self.changes

    def to_json(self, include_diffs: bool) -> dict:
        payload = {
            "tool": self.tool,
            "mode": self.mode,
            "ok": self.ok,
            "summary": {
                "files_scanned": self.files_scanned,
                "files_changed": len(self.changed_files),
                "changes": len(self.changes),
                "issues": len(self.issues),
            },
            "changed_files": self.changed_files,
            "changes": [asdict(change) for change in self.changes],
            "issues": [asdict(issue) for issue in self.issues],
        }
        if include_diffs:
            payload["diffs"] = self.diffs
        return payload


def emit(report: Report, *, as_json: bool, show_diff: bool) -> int:
    """Print `report` and return the process exit code."""
    if as_json:
        json.dump(report.to_json(include_diffs=show_diff), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return EXIT_CLEAN if report.ok else EXIT_PENDING

    verb = "Rewrote" if report.mode == "write" else "Would rewrite"
    for change in report.changes:
        print(f"{change.file}:{change.line}: {change.kind}")
        print(f"    - {change.before}")
        print(f"    + {change.after}")

    if show_diff:
        for diff in report.diffs.values():
            sys.stdout.write(diff)

    for issue in report.issues:
        location = f"{issue.file}:{issue.line}"
        print(f"{location}: needs review ({issue.kind}): {issue.message}")
        if issue.snippet:
            print(f"    {issue.snippet}")

    print(
        f"\n{report.tool}: scanned {report.files_scanned} file(s), "
        f"{verb.lower()} {len(report.changes)} import/op site(s) "
        f"in {len(report.changed_files)} file(s), "
        f"{len(report.issues)} need review.",
        file=sys.stderr,
    )
    if report.mode != "write" and report.changes:
        print("Re-run with --write to apply.", file=sys.stderr)

    return EXIT_CLEAN if report.ok else EXIT_PENDING


def fail(message: str, *, as_json: bool, tool: str) -> int:
    """Report a fatal error in the same shape as a normal run."""
    if as_json:
        json.dump({"tool": tool, "ok": False, "error": message}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"error: {message}", file=sys.stderr)
    return EXIT_ERROR


def display_path(path: Path, root: Optional[Path] = None) -> str:
    """Render `path` relative to the working directory when possible."""
    base = root or Path.cwd()
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)
