use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use eyre::{Context, Result};
use minijinja::{context, Environment};
use rustpython_parser::{ast, Parse};

use kernels_data::config::Build;

use crate::util::{check_or_infer_kernel_dir, parse_build};

fn extract_all(kernel_dir: &Path, module_name: &str) -> Option<Vec<String>> {
    let init_path = ["torch-ext", "tvm-ffi-ext"]
        .iter()
        .map(|ext_dir| {
            kernel_dir
                .join(ext_dir)
                .join(module_name)
                .join("__init__.py")
        })
        .find(|path| path.exists())?;

    let content = fs::read_to_string(init_path).ok()?;
    let stmts = ast::Suite::parse(&content, "<module>").ok()?;

    for stmt in stmts {
        if let ast::Stmt::Assign(assign) = stmt {
            // Check if this is an assignment to __all__
            let is_all = assign.targets.iter().any(
                |target| matches!(target, ast::Expr::Name(name) if name.id.as_str() == "__all__"),
            );

            if is_all {
                // Extract the list elements
                if let ast::Expr::List(list) = assign.value.as_ref() {
                    let names: Vec<String> = list
                        .elts
                        .iter()
                        .filter_map(|elt| {
                            if let ast::Expr::Constant(constant) = elt {
                                if let ast::Constant::Str(s) = &constant.value {
                                    return Some(s.to_string());
                                }
                            }
                            None
                        })
                        .collect();

                    if !names.is_empty() {
                        return Some(names);
                    }
                }
            }
        }
    }

    None
}

fn extract_functions(kernel_dir: &Path, module_name: &str) -> Option<Vec<String>> {
    let names = extract_all(kernel_dir, module_name)?;
    let functions: Vec<String> = names.into_iter().filter(|n| n != "layers").collect();

    if functions.is_empty() {
        None
    } else {
        Some(functions)
    }
}

fn extract_layers(kernel_dir: &Path, module_name: &str) -> Option<Vec<String>> {
    // Only surface layers when the module re-exports the `layers` submodule.
    let names = extract_all(kernel_dir, module_name)?;
    if !names.iter().any(|n| n == "layers") {
        return None;
    }

    let layers_init = kernel_dir
        .join("torch-ext")
        .join(module_name)
        .join("layers")
        .join("__init__.py");

    let content = fs::read_to_string(&layers_init).ok()?;
    let stmts = ast::Suite::parse(&content, "<module>").ok()?;

    let classes: Vec<String> = stmts
        .into_iter()
        .filter_map(|stmt| match stmt {
            ast::Stmt::ClassDef(class_def) => Some(class_def.name.to_string()),
            _ => None,
        })
        .collect();

    if classes.is_empty() {
        None
    } else {
        Some(classes)
    }
}

const CARD_TEMPLATE: &str = include_str!("init/templates/CARD.md");

fn render_card(build: &Build, kernel_dir: &Path) -> Result<String> {
    let mut strip_env = Environment::new();
    strip_env.set_trim_blocks(true);
    strip_env.set_lstrip_blocks(true);
    strip_env
        .add_template_owned("card_raw", CARD_TEMPLATE.to_owned())
        .wrap_err("Cannot load embedded card template")?;
    let card_template = strip_env
        .get_template("card_raw")
        .wrap_err("Cannot get embedded card template")?
        .render(())
        .wrap_err("Cannot strip raw markers from card template")?;

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.add_template_owned("card", card_template)
        .wrap_err("Cannot load card template")?;

    let repo_id = build.repo_id().ok_or(eyre::eyre!(
        "Cannot fill card template because `repo-id` is not specified in `[general.hub]`"
    ))?;
    let module_name = build.general.name.python_name();
    let functions = extract_functions(kernel_dir, &module_name);
    let layers = extract_layers(kernel_dir, &module_name);
    let has_benchmark = kernel_dir.join("benchmarks").join("benchmark.py").exists();

    env.get_template("card")
        .wrap_err("Cannot get card template")?
        .render(context! {
            repo_id => repo_id,
            version => build.general.version,
            functions => functions,
            layers => layers,
            has_benchmark => has_benchmark,
            upstream => build.general.upstream.as_ref().map(|u| u.to_string()),
            license => build.general.license.to_lowercase(),
        })
        .wrap_err("Cannot render card template")
}

pub fn fill_card(kernel_dir: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let build = parse_build(&kernel_dir)?;
    let content = render_card(&build, &kernel_dir)?;

    match output {
        Some(path) => {
            fs::write(&path, &content)
                .wrap_err_with(|| format!("Cannot write `{}`", path.display()))?;
            eprintln!("Generated {}", path.display());
        }
        None => {
            io::stdout()
                .write_all(content.as_bytes())
                .wrap_err("Cannot write to stdout")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::util::parse_build;

    #[test]
    fn test_render_card_uses_embedded_template_with_version() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        // Note: deliberately no committed `CARD.md` in the kernel dir. The card is
        // rendered from the embedded template, so a stale/absent committed file
        // does not affect the output.
        fs::write(
            kernel_dir.join("build.toml"),
            r#"[general]
name = "test-kernel"
license = "Apache-2.0"
backends = ["cuda"]
version = 3

[general.hub]
repo-id = "test/kernel"

[torch]
"#,
        )
        .unwrap();

        let module_dir = kernel_dir.join("torch-ext").join("test_kernel");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(module_dir.join("__init__.py"), r#"__all__ = ["my_func"]"#).unwrap();

        let build = parse_build(kernel_dir).unwrap();
        let card = render_card(&build, kernel_dir).unwrap();

        assert!(
            card.contains(r#"get_kernel("test/kernel", version=3)"#),
            "usage snippet should include the version argument, card was:\n{card}"
        );
    }

    /// Regression test for version-less cards on the Hub: a kernel may carry a
    /// stale committed `CARD.md` (e.g. the pre-#544 template without `version=`).
    /// The render must ignore it and use the canonical embedded template.
    #[test]
    fn test_render_card_ignores_stale_committed_card() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        fs::write(
            kernel_dir.join("build.toml"),
            r#"[general]
name = "test-kernel"
license = "Apache-2.0"
backends = ["cuda"]
version = 3

[general.hub]
repo-id = "test/kernel"

[torch]
"#,
        )
        .unwrap();

        let module_dir = kernel_dir.join("torch-ext").join("test_kernel");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(module_dir.join("__init__.py"), r#"__all__ = ["my_func"]"#).unwrap();

        // A stale committed template with the legacy, version-less snippet and a
        // marker string. The marker must not leak into the rendered card.
        fs::write(
            kernel_dir.join("CARD.md"),
            "STALE_COMMITTED_MARKER\nkernel_module = get_kernel(\"{{ repo_id }}\")\n",
        )
        .unwrap();

        let build = parse_build(kernel_dir).unwrap();
        let card = render_card(&build, kernel_dir).unwrap();

        assert!(
            card.contains(r#"get_kernel("test/kernel", version=3)"#),
            "should render the canonical (versioned) snippet, card was:\n{card}"
        );
        assert!(
            !card.contains("STALE_COMMITTED_MARKER"),
            "the committed CARD.md must not be read, card was:\n{card}"
        );
    }

    #[test]
    fn test_extract_functions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = ["func_a", "func_b"]"#,
        )
        .unwrap();

        assert_eq!(
            extract_functions(kernel_dir, "test_module"),
            Some(vec!["func_a".to_owned(), "func_b".to_owned()])
        );
    }

    #[test]
    fn test_extract_functions_multiline() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = [
    "func_a",
    "func_b",
    "func_c",
]"#,
        )
        .unwrap();

        assert_eq!(
            extract_functions(kernel_dir, "test_module"),
            Some(vec![
                "func_a".to_owned(),
                "func_b".to_owned(),
                "func_c".to_owned()
            ])
        );
    }

    #[test]
    fn test_extract_functions_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        assert_eq!(extract_functions(temp_dir.path(), "missing"), None);
    }

    #[test]
    fn test_extract_functions_excludes_layers() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = ["layers", "func_a"]"#,
        )
        .unwrap();

        assert_eq!(
            extract_functions(kernel_dir, "test_module"),
            Some(vec!["func_a".to_owned()])
        );
    }

    #[test]
    fn test_extract_functions_only_layers() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(module_dir.join("__init__.py"), r#"__all__ = ["layers"]"#).unwrap();

        assert_eq!(extract_functions(kernel_dir, "test_module"), None);
    }

    #[test]
    fn test_extract_layers() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = ["layers", "func_a"]"#,
        )
        .unwrap();

        let layers_dir = module_dir.join("layers");
        fs::create_dir_all(&layers_dir).unwrap();
        fs::write(
            layers_dir.join("__init__.py"),
            r#"
import torch.nn as nn


class ReLU(nn.Module):
    pass


class Softmax(nn.Module):
    pass
"#,
        )
        .unwrap();

        assert_eq!(
            extract_layers(kernel_dir, "test_module"),
            Some(vec!["ReLU".to_owned(), "Softmax".to_owned()])
        );
    }

    #[test]
    fn test_extract_layers_not_in_all() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(module_dir.join("__init__.py"), r#"__all__ = ["func_a"]"#).unwrap();

        let layers_dir = module_dir.join("layers");
        fs::create_dir_all(&layers_dir).unwrap();
        fs::write(layers_dir.join("__init__.py"), r#"class ReLU: pass"#).unwrap();

        assert_eq!(extract_layers(kernel_dir, "test_module"), None);
    }

    #[test]
    fn test_extract_layers_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = ["layers", "func_a"]"#,
        )
        .unwrap();

        assert_eq!(extract_layers(kernel_dir, "test_module"), None);
    }
}
