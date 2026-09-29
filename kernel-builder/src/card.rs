use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use eyre::{bail, Context, Result};
use kernels_common::config::Build;
use minijinja::{context, Environment};
use rustpython_parser::{ast, Parse};

use crate::util::check_or_infer_kernel_dir;

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
                        // Private exports should not appear in generated cards,
                        // even when they are explicitly listed in __all__.
                        .filter(|name| !name.starts_with('_'))
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

    // `from . import layers` resolves to either a `layers/` package or a flat
    // `layers.py` module, so accept whichever the kernel provides.
    let module_dir = kernel_dir.join("torch-ext").join(module_name);
    let layers_pkg = module_dir.join("layers").join("__init__.py");
    let layers_mod = module_dir.join("layers.py");
    let layers_path = [layers_pkg, layers_mod]
        .into_iter()
        .find(|path| path.exists())?;

    let content = fs::read_to_string(&layers_path).ok()?;
    let stmts = ast::Suite::parse(&content, "<module>").ok()?;

    let classes: Vec<String> = stmts
        .into_iter()
        .filter_map(|stmt| match stmt {
            ast::Stmt::ClassDef(class_def) if !class_def.name.starts_with('_') => {
                Some(class_def.name.to_string())
            }
            _ => None,
        })
        .collect();

    if classes.is_empty() {
        None
    } else {
        Some(classes)
    }
}

fn render_card(build: &Build, kernel_dir: &Path) -> Result<String> {
    let card_template_path = kernel_dir.join("CARD.md");
    if !card_template_path.exists() {
        bail!(
            "CARD.md template not found at `{}`",
            card_template_path.display()
        );
    }

    let template_content = fs::read_to_string(&card_template_path)
        .wrap_err_with(|| format!("Cannot read `{}`", card_template_path.display()))?;

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.add_template_owned("card", template_content)
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
            upstream => build.general.upstream.iter().map(|u| u.as_url().to_string()).collect::<Vec<_>>(),
            source => build.general.source.as_ref().map(|u| u.as_url().to_string()),
            license => build.general.license.to_lowercase(),
        })
        .wrap_err("Cannot render card template")
}

pub fn fill_card(kernel_dir: Option<PathBuf>, output: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let build = Build::open(&kernel_dir)?;
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

    #[test]
    fn test_render_card_upstreams() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(
            temp.path().join("CARD.md"),
            Environment::new()
                .render_str(include_str!("init/templates/CARD.md"), context! {})
                .unwrap(),
        )
        .unwrap();
        let urls = [
            "https://github.com/ronghanghu/torch_generic_nms",
            "https://github.com/ronghanghu/cc_torch",
        ];
        for upstream in [
            "".to_owned(),
            format!("upstream = [{:?}]", urls[0]),
            format!("upstream = {urls:?}"),
        ] {
            fs::write(
                temp.path().join("build.toml"),
                format!(
                    r#"
[general]
name = "cv-utils"
version = 1
edition = 6
license = "MIT"
backends = ["cpu"]
{upstream}
[general.hub]
repo-id = "kernels-community/cv-utils"
[torch-noarch]
"#
                ),
            )
            .unwrap();
            let build = Build::open(temp.path()).unwrap();
            let card = render_card(&build, temp.path()).unwrap();
            assert_eq!(
                card.contains("## Upstream"),
                !build.general.upstream.is_empty()
            );
            for url in &build.general.upstream {
                assert!(card.contains(&format!("- {url}\n")), "{card}");
            }
        }
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
    fn test_extract_functions_excludes_private_exports() {
        for ext_dir in ["torch-ext", "tvm-ffi-ext"] {
            let temp_dir = tempfile::tempdir().unwrap();
            let kernel_dir = temp_dir.path();
            let module_dir = kernel_dir.join(ext_dir).join("test_module");
            fs::create_dir_all(&module_dir).unwrap();
            fs::write(
                module_dir.join("__init__.py"),
                r#"__all__ = ["_internal", "func_a", "__private", "layers", "func_b"]"#,
            )
            .unwrap();

            assert_eq!(
                extract_functions(kernel_dir, "test_module"),
                Some(vec!["func_a".to_owned(), "func_b".to_owned()])
            );

            fs::write(
                module_dir.join("__init__.py"),
                r#"__all__ = ["_internal", "__private"]"#,
            )
            .unwrap();

            assert_eq!(extract_functions(kernel_dir, "test_module"), None);
        }
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
    fn test_extract_layers_flat_module() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let module_dir = kernel_dir.join("torch-ext").join("test_module");
        fs::create_dir_all(&module_dir).unwrap();
        fs::write(
            module_dir.join("__init__.py"),
            r#"__all__ = ["layers", "func_a"]"#,
        )
        .unwrap();

        // Layers exposed via a flat `layers.py` module instead of a `layers/`
        // package, as instructed by the docs' `from . import layers` convention.
        fs::write(
            module_dir.join("layers.py"),
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
    fn test_extract_layers_excludes_private_classes() {
        for layers_file in ["layers.py", "layers/__init__.py"] {
            let temp_dir = tempfile::tempdir().unwrap();
            let kernel_dir = temp_dir.path();
            let module_dir = kernel_dir.join("torch-ext").join("test_module");
            let layers_path = module_dir.join(layers_file);
            fs::create_dir_all(layers_path.parent().unwrap()).unwrap();
            fs::write(module_dir.join("__init__.py"), r#"__all__ = ["layers"]"#).unwrap();
            fs::write(
                &layers_path,
                "class _BaseLayer: pass\nclass ReLU(_BaseLayer): pass\nclass __Private: pass\n",
            )
            .unwrap();

            assert_eq!(
                extract_layers(kernel_dir, "test_module"),
                Some(vec!["ReLU".to_owned()])
            );

            fs::write(&layers_path, "class _BaseLayer: pass\n").unwrap();

            assert_eq!(extract_layers(kernel_dir, "test_module"), None);
        }
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
