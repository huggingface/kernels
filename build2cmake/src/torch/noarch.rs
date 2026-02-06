use std::path::PathBuf;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::{
    config::{Backend, Build, General, Torch},
    fileset::FileSet,
    torch::{
        common::{write_compat_py, write_metadata},
        kernel_ops_identifier,
    },
};

pub fn write_torch_ext_noarch(
    env: &Environment,
    build: &Build,
    target_dir: PathBuf,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let mut file_set = FileSet::default();

    let ops_name = kernel_ops_identifier(&target_dir, &build.general.python_name(), ops_id);

    write_compat_py(&mut file_set)?;
    write_ops_py(env, &build.general.python_name(), &ops_name, &mut file_set)?;
    write_pyproject_toml(env, build.torch.as_ref(), &build.general, &mut file_set)?;
    write_metadata(&build.general, &mut file_set)?;

    Ok(file_set)
}

fn write_ops_py(
    env: &Environment,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("torch-ext");
    path.push(name);
    path.push("_ops.py");
    let writer = file_set.entry(path);

    env.get_template("noarch/_ops.py")
        .wrap_err("Cannot get noarch _ops.py template")?
        .render_to_write(
            context! {
                ops_name => ops_name,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

fn write_pyproject_toml(
    env: &Environment,
    torch: Option<&Torch>,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

    let name = &general.name;
    let data_globs = torch.and_then(|torch| {
        torch
            .data_extensions()
            .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "))
    });

    // Common python dependencies (no backend-specific ones)
    let python_dependencies = itertools::process_results(general.python_depends(), |iter| {
        iter.map(|d| format!("\"{d}\"")).join(", ")
    })?;

    // Collect backend-specific dependencies for all backends
    let mut backend_dependencies = Vec::new();
    for backend in &Backend::all() {
        let deps = itertools::process_results(general.backend_python_depends(*backend), |iter| {
            iter.map(|d| format!("\"{d}\"")).collect::<Vec<_>>()
        })?;

        if !deps.is_empty() {
            backend_dependencies.push((backend.to_string(), deps));
        }
    }

    env.get_template("noarch/pyproject.toml")
        .wrap_err("Cannot get noarch pyproject.toml template")?
        .render_to_write(
            context! {
                data_globs => data_globs,
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
                name => name,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}
