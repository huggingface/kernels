import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fix_op_definitions  # noqa: E402


def build(root: Path, files: dict) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def run(path: Path, *extra, write=True):
    argv = [str(path), *extra]
    if write:
        argv.append("--write")
    return fix_op_definitions.main(argv)


def check(path: Path, *extra):
    return fix_op_definitions.process(
        [path], package_root=None, write=False, exclude=list(extra)
    )


def test_custom_op_namespace_is_replaced_and_import_added(tmp_path):
    pkg = build(
        tmp_path / "relu",
        {
            "__init__.py": "",
            "_ops.py": "ops = None\n",
            "ops.py": (
                "import torch\n"
                "\n"
                '@torch.library.custom_op("relu::relu_fwd", mutates_args=())\n'
                "def relu_fwd(x):\n"
                "    return x\n"
            ),
        },
    )
    assert run(pkg) == 0
    assert (pkg / "ops.py").read_text() == (
        "import torch\n"
        "from ._ops import add_op_namespace_prefix\n"
        "\n"
        '@torch.library.custom_op(add_op_namespace_prefix("relu_fwd"), mutates_args=())\n'
        "def relu_fwd(x):\n"
        "    return x\n"
    )


def test_op_name_and_added_import_are_separate_changes(tmp_path):
    # Adding the `_ops` import is its own edit, so it gets its own entry.
    pkg = build(
        tmp_path / "relu",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n"
                "\n"
                '@torch.library.custom_op("relu::relu_fwd", mutates_args=())\n'
                "def relu_fwd(x):\n"
                "    return x\n"
            ),
        },
    )
    report = check(pkg)
    assert [c.kind for c in report.changes] == ["op-name", "import"]
    assert report.changes[1].before == ""
    assert report.changes[1].after == "from ._ops import add_op_namespace_prefix"
    assert report.to_json(include_diffs=False)["summary"]["changes"] == 2


def test_unprefixed_op_name_is_wrapped(tmp_path):
    pkg = build(
        tmp_path / "relu",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n"
                "\n"
                '@torch.library.custom_op("_flash_attn_forward", mutates_args=(), '
                'device_types="cuda")\n'
                "def fwd(x):\n"
                "    return x\n"
            ),
        },
    )
    assert run(pkg) == 0
    text = (pkg / "ops.py").read_text()
    assert 'add_op_namespace_prefix("_flash_attn_forward")' in text


def test_register_fake_imported_from_torch_library(tmp_path):
    pkg = build(
        tmp_path / "moe",
        {
            "__init__.py": "",
            "ops.py": (
                "from torch.library import register_fake\n"
                "\n"
                '@register_fake("moe::single_marlin_gemm_moe")\n'
                "def fake(x):\n"
                "    return x\n"
            ),
        },
    )
    assert run(pkg) == 0
    text = (pkg / "ops.py").read_text()
    assert 'register_fake(add_op_namespace_prefix("single_marlin_gemm_moe"))' in text
    assert "from ._ops import add_op_namespace_prefix" in text


def test_nested_module_gets_correct_relative_ops_import(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "a/__init__.py": "",
            "a/b/__init__.py": "",
            "a/b/ops.py": (
                'import torch\n\n@torch.library.custom_op("x::y", mutates_args=())\ndef y(t):\n    return t\n'
            ),
        },
    )
    assert run(pkg) == 0
    assert (
        "from ..._ops import add_op_namespace_prefix"
        in (pkg / "a/b/ops.py").read_text()
    )


def test_already_prefixed_is_left_alone(tmp_path):
    original = (
        "from ._ops import add_op_namespace_prefix\n"
        "import torch\n"
        "\n"
        '@torch.library.custom_op(add_op_namespace_prefix("relu_fwd"), mutates_args=())\n'
        "def relu_fwd(x):\n"
        "    return x\n"
    )
    pkg = build(tmp_path / "relu", {"__init__.py": "", "ops.py": original})
    assert run(pkg, write=False) == 0
    assert (pkg / "ops.py").read_text() == original


def test_torch_library_define_is_rewritten(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                'import torch\n\ntorch.library.define("ns::myop(Tensor x) -> Tensor")\n'
            ),
        },
    )
    assert run(pkg) == 0
    assert (
        'add_op_namespace_prefix("myop(Tensor x) -> Tensor")'
        in (pkg / "ops.py").read_text()
    )


def test_library_method_define_is_not_touched(tmp_path):
    original = 'import torch\n\nlib = torch.library.Library("ns", "FRAGMENT")\nlib.define("myop() -> ()")\n'
    pkg = build(tmp_path / "mykernel", {"__init__.py": "", "ops.py": original})
    report = check(pkg)
    assert not report.changes
    assert [issue.kind for issue in report.issues] == ["hardcoded-library-namespace"]


def test_custom_op_def_methods_are_not_touched(tmp_path):
    # `register_autograd`/`register_fake` on the object returned by
    # `custom_op` take a function, not an op name.
    original = (
        "import torch\n"
        "\n"
        "@torch.library.custom_op(add_op_namespace_prefix('silu'), mutates_args=())\n"
        "def _silu(x):\n"
        "    return x\n"
        "\n"
        "_silu.register_autograd(backward, setup_context=setup_context)\n"
        "\n"
        "@_silu.register_fake\n"
        "def _(x):\n"
        "    return x\n"
    )
    pkg = build(tmp_path / "mykernel", {"__init__.py": "", "ops.py": original})
    report = check(pkg)
    assert report.changes == []
    assert report.issues == []


def test_aliased_torch_library_import_is_matched(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "from torch.library import custom_op as _custom_op\n"
                "\n"
                '@_custom_op("a::b", mutates_args=())\n'
                "def b(x):\n"
                "    return x\n"
            ),
        },
    )
    assert run(pkg) == 0
    assert 'add_op_namespace_prefix("b")' in (pkg / "ops.py").read_text()


def _kernel_with_op(root: Path, build_toml: str) -> Path:
    """A kernel-builder project laid out as torch-ext/<name>."""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "build.toml").write_text(build_toml)
    return build(
        root / "torch-ext" / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n"
                "\n"
                '@torch.library.custom_op("a::b", mutates_args=())\n'
                "def b(x):\n"
                "    return x\n"
            ),
        },
    )


def test_aot_and_jit_kernels_use_the_same_helper(tmp_path):
    # `torch` (AOT) and `torch-noarch` (JIT) both generate
    # `add_op_namespace_prefix`, so the tool behaves identically.
    for name, section in [("aot", "[torch]\n"), ("jit", "[torch-noarch]\n")]:
        pkg = _kernel_with_op(
            tmp_path / name, f'[general]\nname = "mykernel"\n{section}'
        )
        assert run(pkg) == 0
        text = (pkg / "ops.py").read_text()
        assert "from ._ops import add_op_namespace_prefix" in text
        assert 'add_op_namespace_prefix("b")' in text


def test_tvm_ffi_kernel_uses_its_own_helper(tmp_path):
    # tvm-ffi's generated _ops exposes `torch_add_op_namespace_prefix` and
    # has no `add_op_namespace_prefix`, so importing the latter would break.
    pkg = _kernel_with_op(
        tmp_path / "k", '[general]\nname = "mykernel"\n[tvm-ffi]\nsrc = []\n'
    )
    assert run(pkg) == 0
    text = (pkg / "ops.py").read_text()
    assert "from ._ops import torch_add_op_namespace_prefix" in text
    assert 'torch_add_op_namespace_prefix("b")' in text
    assert "import add_op_namespace_prefix" not in text


def test_tvm_ffi_helper_is_recognized_as_already_prefixed(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n"
                "from ._ops import torch_add_op_namespace_prefix\n"
                "\n"
                "@torch.library.custom_op(torch_add_op_namespace_prefix('b'), mutates_args=())\n"
                "def b(x):\n"
                "    return x\n"
            ),
        },
    )
    report = check(pkg)
    assert report.changes == []
    assert report.issues == []


def test_helper_name_can_be_overridden(tmp_path):
    pkg = _kernel_with_op(
        tmp_path / "k", '[general]\nname = "mykernel"\n[torch]\nsrc = []\n'
    )
    assert run(pkg, "--helper-name", "torch_add_op_namespace_prefix") == 0
    assert "torch_add_op_namespace_prefix" in (pkg / "ops.py").read_text()


def test_non_literal_op_name_is_reported(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n\nname = 'x::y'\n\n@torch.library.register_fake(name)\ndef fake(t):\n    return t\n"
            ),
        },
    )
    report = check(pkg)
    assert not report.changes
    assert [issue.kind for issue in report.issues] == ["non-literal-op-name"]


def test_fallback_import_antipattern_is_reported(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "try:\n"
                "    from ._ops import add_op_namespace_prefix\n"
                "except ImportError:\n"
                "    def add_op_namespace_prefix(name):\n"
                "        return name\n"
            ),
        },
    )
    kinds = {issue.kind for issue in check(pkg).issues}
    assert "fallback-import" in kinds


def test_redefined_helper_is_reported(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "def add_op_namespace_prefix(name):\n    return f'my_kernel::{name}'\n"
            ),
        },
    )
    assert [issue.kind for issue in check(pkg).issues] == ["rewrapped-helper"]


def test_generated_ops_module_is_skipped(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "_ops.py": (
                "def add_op_namespace_prefix(op_name):\n    return f'ns::{op_name}'\n"
            ),
        },
    )
    report = check(pkg)
    assert report.issues == []
    assert report.files_scanned == 1  # __init__.py only


def test_check_mode_does_not_write(tmp_path):
    original = 'import torch\n\n@torch.library.custom_op("a::b", mutates_args=())\ndef b(x):\n    return x\n'
    pkg = build(tmp_path / "mykernel", {"__init__.py": "", "ops.py": original})
    assert run(pkg, write=False) == 1
    assert (pkg / "ops.py").read_text() == original


def test_import_is_added_only_once_for_multiple_ops(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                "import torch\n"
                "\n"
                '@torch.library.custom_op("a::b", mutates_args=())\n'
                "def b(x):\n"
                "    return x\n"
                "\n"
                '@torch.library.register_fake("a::b")\n'
                "def b_fake(x):\n"
                "    return x\n"
            ),
        },
    )
    assert run(pkg) == 0
    text = (pkg / "ops.py").read_text()
    assert text.count("from ._ops import add_op_namespace_prefix") == 1
    assert text.count('add_op_namespace_prefix("b")') == 2


def test_docstring_only_module_gets_import_after_docstring(tmp_path):
    pkg = build(
        tmp_path / "mykernel",
        {
            "__init__.py": "",
            "ops.py": (
                '"""Docs."""\n\n@torch.library.custom_op("a::b", mutates_args=())\ndef b(x):\n    return x\n'
            ),
        },
    )
    assert run(pkg) == 0
    lines = (pkg / "ops.py").read_text().splitlines()
    assert lines[0] == '"""Docs."""'
    assert lines[1] == "from ._ops import add_op_namespace_prefix"
