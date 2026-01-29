use std::io::Write;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, General, Torch};
use crate::metadata::Metadata;
use crate::FileSet;

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

pub fn render_binding(
    env: &Environment,
    torch: &Torch,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("torch-binding.cmake")
        .wrap_err("Cannot get Torch binding template")?
        .render_to_write(
            context! {
                includes => torch.include.as_ref().map(prefix_and_join_includes),
                name => name,
                src => torch.src
            },
            &mut *write,
        )
        .wrap_err("Cannot render Torch binding template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_extension(
    env: &Environment,
    name: &str,
    ops_name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("torch-extension.cmake")
        .wrap_err("Cannot get Torch extension template")?
        .render_to_write(
            context! {
                name => name,
                ops_name => ops_name,
                platform => std::env::consts::OS,
            },
            &mut *write,
        )
        .wrap_err("Cannot render Torch extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}
