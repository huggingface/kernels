"""Tests for the JSON/exit-code contract that agents consume.

The shape of the report is a public interface: an agent decides whether a
conversion is finished from `ok` and the exit code, and reads `changes` to
see what happened. These tests pin that contract down.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _report import EXIT_CLEAN, EXIT_PENDING, Change, Issue, Report, emit  # noqa: E402


def change(file="a.py", line=1, kind="op-name"):
    return Change(file=file, line=line, kind=kind, before="x", after="y")


def issue(file="a.py", kind="non-literal-op-name"):
    return Issue(file=file, line=1, kind=kind, message="m")


def test_summary_counts_match_the_arrays():
    report = Report(
        tool="t",
        mode="check",
        files_scanned=3,
        changes=[change(), change(line=2)],
        issues=[issue()],
    )
    payload = report.to_json(include_diffs=False)
    assert payload["summary"]["changes"] == len(payload["changes"]) == 2
    assert payload["summary"]["issues"] == len(payload["issues"]) == 1
    assert payload["summary"]["files_scanned"] == 3


def test_changed_files_are_deduplicated_in_order():
    report = Report(
        tool="t",
        mode="check",
        changes=[change(file="b.py"), change(file="a.py"), change(file="b.py", line=9)],
    )
    assert report.changed_files == ["b.py", "a.py"]
    assert report.to_json(include_diffs=False)["summary"]["files_changed"] == 2


def test_ok_is_false_while_changes_are_pending():
    report = Report(tool="t", mode="check", changes=[change()])
    assert report.ok is False


def test_ok_is_true_once_changes_are_applied():
    report = Report(tool="t", mode="write", changes=[change()])
    assert report.ok is True


def test_issues_keep_ok_false_even_after_write():
    # The point of the changes/issues split: `--write` cannot clear an issue,
    # so exit 0 after --write really does mean "nothing left to do".
    report = Report(tool="t", mode="write", changes=[change()], issues=[issue()])
    assert report.ok is False


def test_ok_is_true_for_a_clean_tree():
    assert Report(tool="t", mode="check").ok is True


def test_exit_codes_follow_ok(capsys):
    assert (
        emit(Report(tool="t", mode="check"), as_json=True, show_diff=False)
        == EXIT_CLEAN
    )
    capsys.readouterr()
    pending = Report(tool="t", mode="check", changes=[change()])
    assert emit(pending, as_json=True, show_diff=False) == EXIT_PENDING


def test_json_mode_writes_only_json_to_stdout(capsys):
    report = Report(tool="t", mode="check", files_scanned=1, changes=[change()])
    emit(report, as_json=True, show_diff=False)
    captured = capsys.readouterr()
    # Must parse as a single object: agents pipe this straight into a parser.
    assert json.loads(captured.out)["summary"]["changes"] == 1
    assert captured.err == ""


def test_diffs_are_omitted_unless_requested():
    report = Report(tool="t", mode="check", diffs={"a.py": "--- a\n+++ b\n"})
    assert "diffs" not in report.to_json(include_diffs=False)
    assert report.to_json(include_diffs=True)["diffs"] == {"a.py": "--- a\n+++ b\n"}
