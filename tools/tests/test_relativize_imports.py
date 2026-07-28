import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import relativize_imports  # noqa: E402


def build(root: Path, files: dict) -> Path:
    """Materialize a package tree from a `{relative path: source}` mapping."""
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def run(root: Path, *extra, write=True):
    argv = [str(root), *extra]
    if write:
        argv.append("--write")
    return relativize_imports.main(argv)


def test_from_import_at_package_root(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "from mykernel.utils import helper\n",
            "utils.py": "def helper():\n    pass\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "__init__.py").read_text() == "from .utils import helper\n"


def test_from_import_in_nested_module(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils.py": "",
            "a/__init__.py": "",
            "a/b/__init__.py": "",
            "a/b/mod.py": "from mykernel.utils import helper\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "a/b/mod.py").read_text() == "from ...utils import helper\n"


def test_sibling_import_uses_single_dot(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "a/__init__.py": "",
            "a/one.py": "",
            "a/two.py": "from mykernel.a.one import thing\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "a/two.py").read_text() == "from .one import thing\n"


def test_package_root_import(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "a/__init__.py": "",
            "a/mod.py": "from mykernel import thing\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "a/mod.py").read_text() == "from .. import thing\n"


def test_dotted_import_renames_references(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils/__init__.py": "",
            "utils/helpers.py": "",
            "mod.py": (
                "import mykernel.utils.helpers\n\ndef f():\n    return mykernel.utils.helpers.run(1)\n"
            ),
        },
    )
    assert run(pkg) == 0
    assert (pkg / "mod.py").read_text() == (
        "from .utils import helpers\n\ndef f():\n    return helpers.run(1)\n"
    )


def test_dotted_import_reports_import_and_reference_changes(tmp_path):
    # Rewriting the statement and renaming each reference it stranded are
    # separate edits, so each gets its own entry.
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils/__init__.py": "",
            "utils/helpers.py": "",
            "mod.py": (
                "import mykernel.utils.helpers\n\nmykernel.utils.helpers.run(1)\n"
            ),
        },
    )
    report = relativize_imports.process(
        pkg,
        relativize_imports.ModuleResolver(pkg, package_name="mykernel", module_map={}),
        write=False,
        exclude=[],
    )
    assert [c.kind for c in report.changes] == ["import", "reference"]
    assert report.changes[1].before == "mykernel.utils.helpers"
    assert report.changes[1].after == "helpers"


def test_dotted_import_with_alias_keeps_binding(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils/__init__.py": "",
            "utils/helpers.py": "",
            "mod.py": "import mykernel.utils.helpers as h\n\nh.run()\n",
        },
    )
    assert run(pkg) == 0
    assert (
        pkg / "mod.py"
    ).read_text() == "from .utils import helpers as h\n\nh.run()\n"


def test_external_and_relative_imports_are_untouched(tmp_path):
    original = "import torch\nfrom typing import Optional\nfrom .sibling import thing\nfrom ..other import stuff\n"
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "sibling.py": "", "mod.py": original},
    )
    assert run(pkg) == 0
    assert (pkg / "mod.py").read_text() == original


def test_comments_and_formatting_survive(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils.py": "",
            "mod.py": (
                "from mykernel.utils import (  # keep me\n    alpha,\n    beta,  # and me\n)\n"
            ),
        },
    )
    assert run(pkg) == 0
    assert (pkg / "mod.py").read_text() == (
        "from .utils import (  # keep me\n    alpha,\n    beta,  # and me\n)\n"
    )


def test_vendored_top_level_package_is_auto_detected(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "quack/__init__.py": "",
            "quack/utils.py": "",
            "a/__init__.py": "",
            "a/mod.py": "from quack.utils import thing\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "a/mod.py").read_text() == "from ..quack.utils import thing\n"


def test_module_map_handles_renamed_upstream_package(tmp_path):
    pkg = build(
        tmp_path / "liger_kernels",
        {
            "__init__.py": "",
            "rms_norm.py": "from liger_kernel.ops.utils import calculate\n",
            "utils.py": "",
        },
    )
    assert run(pkg, "--module-map", "liger_kernel.ops=") == 0
    assert (pkg / "rms_norm.py").read_text() == "from .utils import calculate\n"


def test_no_auto_disables_detection(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "utils.py": "", "mod.py": "from mykernel.utils import x\n"},
    )
    assert run(pkg, "--no-auto") == 0
    assert (pkg / "mod.py").read_text() == "from mykernel.utils import x\n"


def test_multi_name_import_is_split(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils.py": "",
            "mod.py": "def f():\n    import torch, mykernel.utils as u\n    return u, torch\n",
        },
    )
    assert run(pkg) == 0
    assert (pkg / "mod.py").read_text() == (
        "def f():\n    import torch\n    from . import utils as u\n    return u, torch\n"
    )


def test_bare_package_import_is_reported_not_rewritten(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "mod.py": "import mykernel\n\nmykernel.thing()\n"},
    )
    assert run(pkg) == 1
    assert (pkg / "mod.py").read_text() == "import mykernel\n\nmykernel.thing()\n"


def test_dangling_reference_is_reported(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "utils/__init__.py": "",
            "utils/helpers.py": "",
            "mod.py": ("import mykernel.utils.helpers\n\nmykernel.other.thing()\n"),
        },
    )
    report = relativize_imports.process(
        pkg,
        relativize_imports.ModuleResolver(pkg, package_name="mykernel", module_map={}),
        write=False,
        exclude=[],
    )
    assert [issue.kind for issue in report.issues] == ["dangling-reference"]


def test_unresolved_target_is_reported(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "mod.py": "from mykernel.missing import thing\n"},
    )
    report = relativize_imports.process(
        pkg,
        relativize_imports.ModuleResolver(pkg, package_name="mykernel", module_map={}),
        write=False,
        exclude=[],
    )
    assert [issue.kind for issue in report.issues] == ["unresolved-target"]


def test_check_mode_does_not_write(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "utils.py": "", "mod.py": "from mykernel.utils import x\n"},
    )
    assert run(pkg, write=False) == 1
    assert (pkg / "mod.py").read_text() == "from mykernel.utils import x\n"


def test_clean_package_exits_zero(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {"__init__.py": "", "utils.py": "", "mod.py": "from .utils import x\n"},
    )
    assert run(pkg, write=False) == 0


@pytest.mark.parametrize(
    "from_parts,target,expected",
    [
        ((), (), "."),
        ((), ("utils",), ".utils"),
        (("a", "b"), (), "..."),
        (("a", "b"), ("a", "b", "c"), ".c"),
        (("a", "b"), ("a", "d"), "..d"),
        (("a",), ("quack", "utils"), "..quack.utils"),
    ],
)
def test_relative_import_shapes(from_parts, target, expected):
    from _ast_utils import relative_import

    assert relative_import(from_parts, target) == expected
