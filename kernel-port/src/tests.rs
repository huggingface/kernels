use crate::{ops, python, recipe, workspace::Workspace};
use std::collections::BTreeMap;
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
    assert!(toml.contains("license = \"MIT\"\nedition = 5\nbackends = [\"cuda\"]"));
    assert!(toml.contains("[torch.stable-abi]\ncuda = \"2.10\"\nrocm = \"2.10\"\n"));
    assert!(toml.contains("[torch]\nsrc = [\n    \"torch-ext/b.cpp\",\n    \"torch-ext/b.h\",\n]"));
    assert!(toml.contains("src = [\n    \"k/a.cu\",\n    \"k/b.h\",\n]"));
    assert!(!toml.contains("skip.py"));
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
        "license = \"MIT\"\nedition = 5\nupstream = \"https://github.com/arogozhnikov/einops.git\"\nbackends = ["
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
