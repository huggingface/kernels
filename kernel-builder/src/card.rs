use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use eyre::{bail, Context, Result};
use minijinja::{context, Environment};
use rustpython_parser::{ast, Parse};

use kernels_data::config::Build;

use crate::util::{check_or_infer_kernel_dir, parse_build};

fn extract_functions(kernel_dir: &Path, module_name: &str) -> Option<Vec<String>> {
    let init_path = kernel_dir
        .join("torch-ext")
        .join(module_name)
        .join("__init__.py");

    let content = fs::read_to_string(&init_path).ok()?;
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
                    let functions: Vec<String> = list
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

                    if !functions.is_empty() {
                        return Some(functions);
                    }
                }
            }
        }
    }

    None
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
    let has_benchmark = kernel_dir.join("benchmarks").join("benchmark.py").exists();

    env.get_template("card")
        .wrap_err("Cannot get card template")?
        .render(context! {
            repo_id => repo_id,
            functions => functions,
            has_benchmark => has_benchmark,
            upstream => build.general.upstream.as_ref().map(|u| u.to_string()),
            license => build.general.license.clone(),
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
}
