use std::path::PathBuf;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, General};
use crate::metadata::Metadata;
use crate::FileSet;

static REGISTRATION_H: &str = include_str!("../templates/registration.h");

pub fn write_pyproject_toml(
    env: &Environment,
    backend: Backend,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

    let python_dependencies = itertools::process_results(
        general
            .python_depends()
            .chain(general.backend_python_depends(backend)),
        |iter| iter.map(|d| format!("\"{d}\"")).join(", "),
    )?;

    env.get_template("pyproject.toml")
        .wrap_err("Cannot get pyproject.toml template")?
        .render_to_write(
            context! {
                python_dependencies => python_dependencies,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

pub fn write_metadata(backend: Backend, general: &General, file_set: &mut FileSet) -> Result<()> {
    let writer = file_set.entry("metadata.json");

    let python_depends = general
        .python_depends()
        .chain(general.backend_python_depends(backend))
        .collect::<Result<Vec<_>>>()?;

    let metadata = Metadata {
        version: general.version,
        license: general.license.clone(),
        python_depends,
    };

    serde_json::to_writer_pretty(writer, &metadata)?;

    Ok(())
}

pub fn prefix_and_join_includes<S>(includes: impl AsRef<[S]>) -> String
where
    S: AsRef<str>,
{
    includes
        .as_ref()
        .iter()
        .map(|include| format!("${{CMAKE_SOURCE_DIR}}/{}", include.as_ref()))
        .collect_vec()
        .join(";")
}

pub fn write_torch_registration_macros(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("torch-ext");
    path.push("registration.h");
    file_set
        .entry(path)
        .extend_from_slice(REGISTRATION_H.as_bytes());

    Ok(())
}
