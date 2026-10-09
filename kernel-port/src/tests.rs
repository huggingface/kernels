use crate::{ops, python, recipe, workspace::Workspace};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[test]
fn recipe_basic_statement() {
    let parsed = recipe::parse("delete in=\".github/**\"  // trailing comment\n").unwrap();
    assert_eq!(parsed.ops.len(), 1);
    assert_eq!(parsed.ops[0].op, "delete");
    assert_eq!(parsed.ops[0].take_args().take("in").unwrap(), ".github/**");
    assert_eq!(parsed.ops[0].line, 1);
}

#[test]
fn replace_defaults_to_one_match() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("a.py".into(), b"before\n".to_vec()),
        ("b.py".into(), b"before\n".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "replace in=\"*.py\" find=\"before\" with=\"after\"\n",
    );
    assert_eq!(ws.get_text("a.py").unwrap(), "after\n");
    assert_eq!(ws.get_text("b.py").unwrap(), "after\n");
}

#[test]
fn recipe_escapes_and_typed_values() {
    let parsed =
        recipe::parse(r#"replace in="f" find="a\nb\t\"q\"\\" with="" count=3 flag=#true"#).unwrap();
    let mut args = parsed.ops[0].take_args();
    assert_eq!(args.take("find").unwrap(), "a\nb\t\"q\"\\");
    assert_eq!(args.take("with").unwrap(), "");
    assert_eq!(args.take_usize("count").unwrap(), 3);
    assert_eq!(args.take("flag").unwrap(), "true");
}

#[test]
fn recipe_multiline_raw_string_boundary_newlines() {
    let parsed = recipe::parse(
        "replace in=\"f\" count=1 with=#\"\"\"\n\nno \\n escape \"quoted\"\n\n\"\"\"# find=\"x\"\n",
    )
    .unwrap();
    let mut args = parsed.ops[0].take_args();
    assert_eq!(args.take("with").unwrap(), "\nno \\n escape \"quoted\"\n");
    assert_eq!(args.take("find").unwrap(), "x");
}

#[test]
fn recipe_multiline_without_trailing_newline() {
    let parsed =
        recipe::parse("expect in=\"f\" count=1 find=#\"\"\"\n\nvoid\nf(int x) {\n\"\"\"#\n")
            .unwrap();
    let mut args = parsed.ops[0].take_args();
    assert_eq!(args.take("find").unwrap(), "\nvoid\nf(int x) {");
}

#[test]
fn recipe_comments_blanks_and_line_numbers() {
    let parsed = recipe::parse("// header\n\nmove from=\"a\" to=\"b\"\n\n// tail\n").unwrap();
    assert_eq!(parsed.ops.len(), 1);
    assert_eq!(parsed.ops[0].line, 3);
}

#[test]
fn recipe_rejects_duplicate_and_unknown_args() {
    let err = recipe::parse("delete in=\"a\" in=\"b\"\n")
        .err()
        .expect("duplicate arguments are rejected")
        .to_string();
    assert!(err.contains("duplicate argument"), "{err}");
    let parsed = recipe::parse("delete in=\"a\" bogus=\"b\"\n").unwrap();
    let mut args = parsed.ops[0].take_args();
    args.take("in").unwrap();
    assert!(args.finish().is_err());
}

#[test]
fn recipe_rejects_foreign_kdl_shapes() {
    // KDL allows these shapes; the recipe language does not.
    for (text, what) in [
        ("delete \"a\"\n", "positional"),
        ("delete in=\"a\" { child; }\n", "children"),
        ("delete in=(glob)\"a\"\n", "type annotation"),
        ("manifest version=1.5\n", "float value"),
    ] {
        assert!(recipe::parse(text).is_err(), "{what} should be rejected");
    }
}

#[test]
fn recipe_version_header_is_consumed_not_run() {
    let parsed = recipe::parse("recipe version=1\ndelete in=\"a\"\n").unwrap();
    assert_eq!(parsed.version, Some(1));
    assert_eq!(parsed.ops.len(), 1);
    assert_eq!(parsed.ops[0].op, "delete");
    assert_eq!(parsed.ops[0].line, 2);
}

#[test]
fn recipe_without_header_has_no_declared_version() {
    let parsed = recipe::parse("delete in=\"a\"\n").unwrap();
    assert_eq!(parsed.version, None);
    assert_eq!(parsed.effective_version(), recipe::VERSION);
}

#[test]
fn recipe_version_rejects_unknown_and_misplaced_headers() {
    let err = |text: &str| recipe::parse(text).err().unwrap().to_string();
    assert!(err("recipe version=99\n").contains("this kernel-port implements version 1"));
    assert!(err("recipe version=0\n").contains("this kernel-port implements version 1"));
    assert!(err("delete in=\"a\"\nrecipe version=1\n").contains("must be the first"));
    assert!(err("recipe\n").contains("requires argument version"));
    assert!(err("recipe version=1 name=\"x\"\n").contains("does not take an argument named"));
}

#[test]
fn relativize_dots_formula() {
    let src = "from pkg.core import base\nfrom pkg.core.base import Base\nfrom pkg.utils import x\nfrom os.path import join\n";
    let (out, n) = python::relativize_source("f.py", src, &["pkg", "core"])
        .unwrap()
        .unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        out,
        "from . import base\nfrom .base import Base\nfrom ..utils import x\nfrom os.path import join\n"
    );
}

#[test]
fn relativize_from_package_root_preserves_full_module_path() {
    let src = "from pkg.ops import base\nfrom pkg.ops.base import Base\nfrom pkg.utils import x\n";
    let (out, n) =
        python::relativize_source_from_root("pkg/ops/core/engine.py", src, &["pkg", "ops", "core"])
            .unwrap()
            .unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        out,
        "from ...ops import base\nfrom ...ops.base import Base\nfrom ...utils import x\n"
    );
}

#[test]
fn relativize_preserves_comments_and_layout() {
    let src = "from pkg.a import (\n    x,  # keep\n)\n";
    let (out, _) = python::relativize_source("f.py", src, &["pkg"])
        .unwrap()
        .unwrap();
    assert_eq!(out, "from .a import (\n    x,  # keep\n)\n");
}

#[test]
fn relativize_untouched_returns_none() {
    let src = "import os\nfrom .a import b\n";
    assert!(
        python::relativize_source("f.py", src, &["pkg"])
            .unwrap()
            .is_none()
    );
}

#[test]
// infer_device is a text prefix of infer_device_arch: each statement has to be
// spliced on its own boundaries.
fn remap_prefix_and_boundary_collision() {
    let src = "from liger_kernel.utils import infer_device\nfrom liger_kernel.utils import infer_device_arch\n";
    let from: python::DottedPath = "liger_kernel.utils".parse().unwrap();
    let to: python::DottedPath = "liger_kernels._liger_utils".parse().unwrap();
    let (out, n) = python::remap_source("f.py", src, &from, &to)
        .unwrap()
        .unwrap();
    assert_eq!(n, 2);
    assert_eq!(
        out,
        "from liger_kernels._liger_utils import infer_device\nfrom liger_kernels._liger_utils import infer_device_arch\n"
    );
}

#[test]
fn remap_does_not_touch_other_prefixes() {
    let src = "from liger_kernel.ops.rms_norm import f\n";
    let from: python::DottedPath = "liger_kernel.utils".parse().unwrap();
    let to: python::DottedPath = "x".parse().unwrap();
    assert!(
        python::remap_source("f.py", src, &from, &to)
            .unwrap()
            .is_none()
    );
}

#[test]
fn splice_refuses_string_literal_duplicates() {
    let src = "from pkg.a import b\ns = \"from pkg.a import b\"\n";
    assert!(python::relativize_source("f.py", src, &["pkg"]).is_err());
}

#[test]
fn imports_inside_functions_are_rewritten() {
    let src = "def f():\n    from pkg.a import b\n    return b\n";
    let (out, _) = python::relativize_source("f.py", src, &["pkg"])
        .unwrap()
        .unwrap();
    assert_eq!(out, "def f():\n    from .a import b\n    return b\n");
}

#[test]
fn ensure_import_appends_after_module_initialization() {
    let src = "\"\"\"docs\"\"\"\n\nVALUE = 1\nfrom .packing import pack\n";
    let (out, n) = python::ensure_import_source("pkg/__init__.py", src, ".", "array_api")
        .unwrap()
        .unwrap();
    assert_eq!(n, 1);
    assert_eq!(
        out,
        "\"\"\"docs\"\"\"\n\nVALUE = 1\nfrom .packing import pack\nfrom . import array_api\n"
    );
}

#[test]
fn ensure_import_recognizes_an_existing_name_in_a_group() {
    let src = "from . import other, array_api  # public modules\n";
    assert!(
        python::ensure_import_source("pkg/__init__.py", src, ".", "array_api")
            .unwrap()
            .is_none()
    );
}

#[test]
fn ensure_import_rejects_duplicate_names_in_one_group() {
    let src = "from . import array_api, array_api\n";
    let err = python::ensure_import_source("pkg/__init__.py", src, ".", "array_api")
        .unwrap_err()
        .to_string();
    assert!(err.contains("satisfied by 2 top-level imports"), "{err}");
}

#[test]
fn ensure_import_ignores_nested_imports_and_preserves_no_final_newline() {
    let src = "def load():\n    from . import array_api\n";
    let (out, _) = python::ensure_import_source("pkg/__init__.py", src, ".", "array_api")
        .unwrap()
        .unwrap();
    assert_eq!(
        out,
        "def load():\n    from . import array_api\nfrom . import array_api\n"
    );

    let (out, _) = python::ensure_import_source("pkg/__init__.py", "VALUE = 1", ".", "array_api")
        .unwrap()
        .unwrap();
    assert_eq!(out, "VALUE = 1\nfrom . import array_api");
}

#[test]
fn ensure_import_recipe_pins_changes() {
    let original = BTreeMap::from([("pkg/__init__.py".into(), b"VALUE = 1\n".to_vec())]);
    let mut ws = Workspace::from_files(original.clone());
    run_recipe(
        &mut ws,
        "ensure_import in=\"pkg/__init__.py\" from=\".\" name=\"array_api\" changes=1\n",
    );
    assert_eq!(
        ws.get_text("pkg/__init__.py").unwrap(),
        "VALUE = 1\nfrom . import array_api\n"
    );

    let mut ws = Workspace::from_files(original);
    let err = run_recipe_err(
        &mut ws,
        "ensure_import in=\"pkg/__init__.py\" from=\".\" name=\"array_api\" changes=0\n",
    );
    assert!(
        err.contains("expected exactly 0 change(s) but made 1"),
        "{err}"
    );
}

fn kernelize(src: &str, binding: &str) -> String {
    python::kernelize_imports_source(
        "tests/test_x.py",
        src,
        "einops",
        binding,
        "kernels-community/einops",
        1,
    )
    .unwrap()
    .unwrap()
    .0
}

#[test]
fn kernelize_imports_binds_the_kernel_once_and_uses_attributes() {
    let src = concat!(
        "import os\n",
        "import einops\n",
        "import einops.layers\n",
        "import einops as eo\n",
        "import einops.layers.torch as torch_layers\n",
        "from einops.parsing import ParsedExpression, _ellipsis as ell  # parsing\n",
        "from . import helpers\n",
        "def load():\n",
        "    from einops.layers.torch import Rearrange as R, Reduce\n",
        "    from einops import array_api as AA\n",
        "    import einops\n",
        "    return R, Reduce, AA\n",
        "class TestX:\n",
        "    def test_m(self):\n",
        "        from einops import rearrange\n",
        "        return rearrange\n",
    );
    let (out, n) = python::kernelize_imports_source(
        "tests/test_x.py",
        src,
        "einops",
        "einops",
        "kernels-community/einops",
        1,
    )
    .unwrap()
    .unwrap();
    assert_eq!(n, 9);
    assert_eq!(
        out,
        concat!(
            "import os\n",
            "import kernels\n",
            "einops = kernels.get_kernel(\"kernels-community/einops\", version=1)\n",
            "eo = einops\n",
            "torch_layers = einops.layers.torch\n",
            "ParsedExpression = einops.parsing.ParsedExpression\n",
            "ell = einops.parsing._ellipsis  # parsing\n",
            "from . import helpers\n",
            "def load():\n",
            "    R, Reduce = einops.layers.torch.Rearrange, einops.layers.torch.Reduce\n",
            "    AA = einops.array_api\n",
            "    return R, Reduce, AA\n",
            "class TestX:\n",
            "    def test_m(self):\n",
            "        rearrange = einops.rearrange\n",
            "        return rearrange\n",
        )
    );
    assert!(!out.contains("importlib"));
    assert!(
        python::absolute_self_imports("tests/test_x.py", &out, "einops")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn kernelize_imports_custom_binding_rebinds_the_package_name() {
    let out = kernelize(
        "\"\"\"docs\"\"\"\nimport os\nfrom einops import _backends\nimport einops\n",
        "_einops",
    );
    assert_eq!(
        out,
        concat!(
            "\"\"\"docs\"\"\"\nimport os\n",
            "import kernels\n",
            "_einops = kernels.get_kernel(\"kernels-community/einops\", version=1)\n",
            "_backends = _einops._backends\n",
            "einops = _einops\n",
        )
    );
}

#[test]
fn kernelize_imports_binds_at_the_top_when_only_nested_imports_exist() {
    let out = kernelize(
        "\"\"\"docs\"\"\"\nfrom __future__ import annotations\nimport os\ndef f():\n    from einops import rearrange\n",
        "einops",
    );
    assert_eq!(
        out,
        concat!(
            "\"\"\"docs\"\"\"\nfrom __future__ import annotations\n",
            "import kernels\n",
            "einops = kernels.get_kernel(\"kernels-community/einops\", version=1)\n",
            "import os\ndef f():\n    rearrange = einops.rearrange\n",
        )
    );
}

#[test]
fn kernelize_imports_keeps_one_statement_when_sharing_a_line() {
    let out = kernelize(
        "import os; from einops import a, b\nif x: from einops import c, d\n",
        "einops",
    );
    assert!(out.contains("import os; a, b = einops.a, einops.b\n"));
    assert!(out.contains("if x: c, d = einops.c, einops.d\n"));
}

#[test]
fn kernelize_imports_uses_one_form_for_repeated_import_text() {
    let out = kernelize(
        "from einops import a, b\nfrom einops import c\ndef f():\n    from einops import a, b\n    from einops import c\n",
        "einops",
    );
    assert!(out.contains("version=1)\na, b = einops.a, einops.b\nc = einops.c\ndef f():\n    a, b = einops.a, einops.b\n    c = einops.c\n"));
}

#[test]
fn kernelize_imports_rejects_unsafe_static_forms() {
    for (src, message) in [
        ("from einops import *\n", "wildcard import"),
        (
            "from einops import (rearrange, reduce)\n",
            "parenthesized import",
        ),
        ("import os, einops\n", "multi-name import"),
        ("import einops; import os\n", "shares its line"),
    ] {
        let err = python::kernelize_imports_source(
            "tests/test_x.py",
            src,
            "einops",
            "einops",
            "kernels-community/einops",
            1,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains(message), "{err}");
    }
}

#[test]
fn mark_tests_decorates_tests_above_existing_decorators() {
    let src = concat!(
        "from doctest import testmod\n",
        "import pytest\n",
        "\n",
        "def helper():\n",
        "    pass\n",
        "\n",
        "# comment\n",
        "def test_a():\n",
        "    def test_nested():\n",
        "        pass\n",
        "\n",
        "@pytest.mark.parametrize(\"x\", [1])\n",
        "async def test_b(x):\n",
        "    pass\n",
        "\n",
        "@pytest.mark.kernels_ci\n",
        "def test_c():\n",
        "    pass\n",
        "\n",
        "class TestD:\n",
        "    def test_e(self):\n",
        "        pass\n",
    );
    let (out, n) = python::mark_tests_source(
        "tests/test_x.py",
        src,
        "kernels_ci",
        &BTreeSet::new(),
        &mut BTreeSet::new(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(n, 3);
    assert!(out.contains("# comment\n@pytest.mark.kernels_ci\ndef test_a():\n    def test_nested"));
    assert!(out.contains(
        "\n@pytest.mark.kernels_ci\n@pytest.mark.parametrize(\"x\", [1])\nasync def test_b"
    ));
    assert!(out.contains("\n\n@pytest.mark.kernels_ci\ndef test_c"));
    assert!(out.contains("\n@pytest.mark.kernels_ci\nclass TestD:\n    def test_e"));
    assert!(out.starts_with("from doctest import testmod\n"));
}

#[test]
fn mark_tests_requires_pytest_import() {
    let err = python::mark_tests_source(
        "tests/test_x.py",
        "def test_a():\n    pass\n",
        "kernels_ci",
        &BTreeSet::new(),
        &mut BTreeSet::new(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("import pytest"), "{err}");
}

#[test]
fn mark_tests_exclude_leaves_named_tests_unmarked() {
    let src = "import pytest\n\ndef test_a():\n    pass\n\nclass TestB:\n    pass\n";
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("tests/test_x.py".into(), src.as_bytes().to_vec()),
        ("tests/test_y.py".into(), src.as_bytes().to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "mark_tests in=\"tests/*.py\" marker=\"kernels_ci\" exclude=\"TestB\" changes=2\n",
    );
    for path in ["tests/test_x.py", "tests/test_y.py"] {
        let out = ws.get_text(path).unwrap();
        assert!(out.contains("@pytest.mark.kernels_ci\ndef test_a"), "{out}");
        assert!(out.contains("\n\nclass TestB"), "{out}");
    }
}

#[test]
fn mark_tests_exclude_rejects_stale_names() {
    let mut ws = Workspace::from_files(BTreeMap::from([(
        "tests/test_x.py".into(),
        b"import pytest\n\ndef test_a():\n    pass\n".to_vec(),
    )]));
    let err = run_recipe_err(
        &mut ws,
        "mark_tests in=\"tests/*.py\" marker=\"kernels_ci\" exclude=\"test_gone\"\n",
    );
    assert!(err.contains("test_gone"), "{err}");
}

#[test]
fn kernelize_imports_recipe_pins_statements() {
    let mut ws = Workspace::from_files(BTreeMap::from([(
        "tests/test_x.py".into(),
        b"from einops import rearrange\n".to_vec(),
    )]));
    run_recipe(
        &mut ws,
        "kernelize_imports in=\"tests/**\" package=\"einops\" kernel=\"kernels-community/einops\" version=1 changes=1\n",
    );
    assert!(
        ws.get_text("tests/test_x.py")
            .unwrap()
            .contains("get_kernel(\"kernels-community/einops\", version=1)")
    );
}

fn run_recipe_err(ws: &mut Workspace, recipe_text: &str) -> String {
    let parsed = recipe::parse(recipe_text).unwrap();
    let inputs = ops::Inputs::default();
    let mut facts = ops::Facts::default();
    for inv in &parsed.ops {
        match ops::build(inv, Path::new(".")).and_then(|op| op.apply(ws, &inputs, &mut facts)) {
            Ok(_) => {}
            Err(e) => return e.to_string(),
        }
    }
    panic!("recipe unexpectedly succeeded");
}

fn run_recipe(ws: &mut Workspace, recipe_text: &str) {
    let parsed = recipe::parse(recipe_text).unwrap();
    let inputs = ops::Inputs::default();
    let mut facts = ops::Facts::default();
    for inv in &parsed.ops {
        let op = ops::build(inv, Path::new(".")).unwrap();
        op.apply(ws, &inputs, &mut facts).unwrap();
    }
}

#[test]
fn manifest_torch_mode() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\" capabilities=\"9.0\"\nmanifest name=\"k\" version=1 license=\"MIT\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("backends = [\"cuda\"]"));
    assert!(toml.contains(
        "[kernel.k]\nbackend = \"cuda\"\ncuda-capabilities = [\"9.0\"]\ndepends = [\"torch\"]\nsrc = [\n    \"k/a.cu\",\n]"
    ));
}

#[test]
fn manifest_kernel_cuda_flags() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\" cuda_flags=\"-O3,--use_fast_math\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    assert!(
        ws.get_text("build.toml")
            .unwrap()
            .contains("cuda-flags = [\n    \"-O3\",\n    \"--use_fast_math\",\n]")
    );
}

#[test]
fn manifest_kernel_cuda_minver() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\" cuda_minver=\"12.9\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    assert!(
        ws.get_text("build.toml")
            .unwrap()
            .contains("backend = \"cuda\"\ncuda-minver = \"12.9\"\ndepends")
    );
}

#[test]
fn manifest_kernel_rocm_archs() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"rocm\" src=\"k/*\" rocm_archs=\"gfx90a,gfx942\"\nmanifest name=\"k\" backends=\"rocm\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    assert!(ws.get_text("build.toml").unwrap().contains(
        "backend = \"rocm\"\ndepends = [\"torch\"]\nrocm-archs = [\n    \"gfx90a\",\n    \"gfx942\",\n]\n"
    ));
}

#[test]
fn manifest_kernel_repeat_src() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\" repeat_src=\"k/a.cu\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert_eq!(toml.matches("\"k/a.cu\"").count(), 2);
}

#[test]
fn manifest_kernel_cuda_flags_preserve_doubled_comma() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\" cuda_flags=\"-O3,--ptxas-options=--verbose,,--warn-on-local-memory-usage\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    assert!(ws.get_text("build.toml").unwrap().contains(
        "cuda-flags = [\n    \"-O3\",\n    \"--ptxas-options=--verbose,--warn-on-local-memory-usage\",\n]"
    ));
}

#[test]
fn manifest_cuda_version_bounds() {
    let mut ws = Workspace::from_files(BTreeMap::new());
    run_recipe(
        &mut ws,
        "manifest name=\"k\" backends=\"cuda\" cuda_minver=\"12.0\" cuda_maxver=\"12.9\" noarch=#true\n",
    );
    assert!(
        ws.get_text("build.toml")
            .unwrap()
            .contains("[general.cuda]\nminver = \"12.0\"\nmaxver = \"12.9\"\n")
    );
}

#[test]
fn glob_lists_keep_brace_alternates_whole() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.h".into(), b"x".to_vec()),
        ("k/a.cpp".into(), b"x".to_vec()),
        ("k/a.py".into(), b"x".to_vec()),
        ("t/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*.{h,cpp},t/*\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"t/*\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("src = [\n    \"k/a.cpp\",\n    \"k/a.h\",\n    \"t/b.cpp\",\n]"));
    assert!(!toml.contains("a.py"));
}

#[test]
fn kernel_accepts_workspace_root_include() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" include=\".\" src=\"k/*.cu\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    assert!(
        ws.get_text("build.toml")
            .unwrap()
            .contains("include = [\".\"]")
    );
}

#[test]
fn manifest_edition_stable_abi_and_multi_glob_src() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("k/b.h".into(), b"x".to_vec()),
        ("k/skip.py".into(), b"x".to_vec()),
        ("torch-ext/b.cpp".into(), b"x".to_vec()),
        ("torch-ext/b.h".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*.cu,k/*.h\"\nmanifest name=\"k\" version=1 license=\"MIT\" edition=5 backends=\"cuda\" torch_src=\"torch-ext/*.cpp,torch-ext/*.h\" stable_abi=\"cuda=2.10,rocm=2.10\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("edition = 5\nlicense = \"MIT\"\nbackends = [\"cuda\"]"));
    assert!(toml.contains("[torch.stable-abi]\ncuda = \"2.10\"\nrocm = \"2.10\"\n"));
    assert!(toml.contains("[torch]\nsrc = [\n    \"torch-ext/b.cpp\",\n    \"torch-ext/b.h\",\n]"));
    assert!(toml.contains("src = [\n    \"k/a.cu\",\n    \"k/b.h\",\n]"));
    assert!(!toml.contains("skip.py"));
}

#[test]
fn manifest_edition_6_tags_kernel_language() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/a.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*.cu\"\nmanifest name=\"k\" version=1 license=\"MIT\" edition=6 backends=\"cuda\" torch_src=\"torch-ext/*.cpp\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("[kernel.k]\nlanguage = \"cpp-cuda\"\n"));
    assert!(!toml.contains("backend = "));
}

#[test]
fn manifest_scalar_stable_abi() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/a.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*\" stable_abi_version=\"2.9\"\n",
    );
    assert!(
        ws.get_text("build.toml")
            .unwrap()
            .contains("[torch]\nstable-abi = \"2.9\"\nsrc = [")
    );
}

#[test]
fn manifest_torch_include() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/a.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*\" torch_include=\"k\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(
        toml.contains("[torch]\ninclude = [\"k\"]\nsrc = ["),
        "{toml}"
    );
}

#[test]
fn manifest_torch_include_must_exist() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/a.cpp".into(), b"x".to_vec()),
    ]));
    let err = run_recipe_err(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*\" torch_include=\"nope\"\n",
    );
    assert!(
        err.contains("include directory \"nope\" contains no files"),
        "{err}"
    );
}

#[test]
fn manifest_torch_pyext() {
    let mut ws = Workspace::from_files(BTreeMap::from([
        ("k/a.cu".into(), b"x".to_vec()),
        ("torch-ext/a.cpp".into(), b"x".to_vec()),
    ]));
    run_recipe(
        &mut ws,
        "kernel name=\"k\" backend=\"cuda\" src=\"k/*\"\nmanifest name=\"k\" backends=\"cuda\" torch_src=\"torch-ext/*\" torch_pyext=\"py,cuh,hpp,h\"\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(
        toml.contains(
            "[torch]\npyext = [\n    \"py\",\n    \"cuh\",\n    \"hpp\",\n    \"h\",\n]\nsrc = ["
        ),
        "{toml}"
    );
}

#[test]
fn manifest_noarch_mode() {
    let mut ws = Workspace::from_files(BTreeMap::new());
    run_recipe(
        &mut ws,
        "manifest name=\"k\" backends=\"cuda,rocm\" noarch=#true\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("backends = [\n    \"cuda\",\n    \"rocm\",\n]"));
    assert!(toml.ends_with("[torch-noarch]\n\n[kernel]\n"));
}

#[test]
fn manifest_upstream_field() {
    let mut ws = Workspace::from_files(BTreeMap::new());
    run_recipe(
        &mut ws,
        "manifest name=\"einops\" version=1 license=\"MIT\" edition=5 upstream=\"https://github.com/arogozhnikov/einops.git\" backends=\"cpu,cuda\" noarch=#true\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains(
        "edition = 5\nlicense = \"MIT\"\nupstream = \"https://github.com/arogozhnikov/einops.git\"\nbackends = ["
    ));
}

#[test]
fn manifest_general_cuda_table() {
    let mut ws = Workspace::from_files(BTreeMap::new());
    run_recipe(
        &mut ws,
        "manifest name=\"k\" version=0 backends=\"cuda\" python_depends=\"einops,tvm-ffi\" cuda_minver=\"12.8\" cuda_python_depends=\"nvidia-cutlass-dsl\" repo_id=\"x/k\" noarch=#true\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("python-depends = [\n    \"einops\",\n    \"tvm-ffi\",\n]\n\n[general.cuda]\nminver = \"12.8\"\npython-depends = [\"nvidia-cutlass-dsl\"]\n\n[general.hub]\n"));
}

#[test]
fn manifest_hub_branch() {
    let mut ws = Workspace::from_files(BTreeMap::new());
    run_recipe(
        &mut ws,
        "manifest name=\"k\" backends=\"cuda\" repo_id=\"x/k\" hub_branch=\"ep-support\" noarch=#true\n",
    );
    let toml = ws.get_text("build.toml").unwrap();
    assert!(toml.contains("[general.hub]\nrepo-id = \"x/k\"\nbranch = \"ep-support\"\n"));
}

#[test]
fn expect_guards() {
    let mut ws = Workspace::from_files(BTreeMap::from([("a.py".into(), b"x = 1\n".to_vec())]));
    run_recipe(
        &mut ws,
        "expect in=\"*.py\" files=1\nexpect in=\"a.py\" find=\"x\" count=1\nexpect in=\"a.py\" find=\"gone\" count=0\n",
    );
    let parsed = recipe::parse("expect in=\"*.py\" files=2\n").unwrap();
    let op = ops::build(&parsed.ops[0], Path::new(".")).unwrap();
    assert!(
        op.apply(&mut ws, &ops::Inputs::default(), &mut ops::Facts::default())
            .is_err()
    );
}

#[test]
fn strip_suffix_is_fully_pinned() {
    let original = BTreeMap::from([
        ("a.h".into(), b"a\n".to_vec()),
        ("b.h".into(), b"b\n".to_vec()),
    ]);
    let mut ws = Workspace::from_files(original.clone());
    run_recipe(&mut ws, "strip_suffix in=\"*.h\" suffix=\"\\n\" files=2\n");
    assert_eq!(ws.get_text("a.h").unwrap(), "a");
    assert_eq!(ws.get_text("b.h").unwrap(), "b");

    let mut ws = Workspace::from_files(original.clone());
    let parsed = recipe::parse("strip_suffix in=\"*.h\" suffix=\"\\n\" files=1\n").unwrap();
    let op = ops::build(&parsed.ops[0], Path::new(".")).unwrap();
    assert!(
        op.apply(&mut ws, &ops::Inputs::default(), &mut ops::Facts::default())
            .unwrap_err()
            .to_string()
            .contains("expected exactly 1 file(s)")
    );

    let mut missing = original;
    missing.insert("b.h".into(), b"b".to_vec());
    let mut ws = Workspace::from_files(missing);
    let parsed = recipe::parse("strip_suffix in=\"*.h\" suffix=\"\\n\" files=2\n").unwrap();
    let op = ops::build(&parsed.ops[0], Path::new(".")).unwrap();
    assert!(
        op.apply(&mut ws, &ops::Inputs::default(), &mut ops::Facts::default())
            .unwrap_err()
            .to_string()
            .contains("does not end with pinned suffix")
    );
    assert_eq!(ws.get_text("a.h").unwrap(), "a\n");
}
