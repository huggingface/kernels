use std::path::PathBuf;

use eyre::{bail, Context, Result};
use itertools::Itertools;
use kernels_common::config::{Build, General, MlxNoarch};
use kernels_common::metadata::Provenance;
use minijinja::{context, Environment};

use crate::pyproject::common::write_metadata_for_backends;
use crate::pyproject::fileset::FileSet;
use crate::pyproject::ops_identifier::KernelIdentifier;

static SETUP_PY: &str = include_str!("../templates/mlx/noarch/setup.py");

pub fn write_mlx_ext_noarch(
    env: &Environment,
    build: &Build,
    kernel_id: &KernelIdentifier,
    provenance: Option<&Provenance>,
) -> Result<FileSet> {
    let mlx_noarch = match build.framework.mlx_noarch() {
        Some(mlx_noarch) => mlx_noarch,
        None => bail!("Build configuration does not have `mlx-noarch` section"),
    };

    let mut file_set = FileSet::default();

    write_ops_py(
        env,
        &build.general.name.python_name(),
        kernel_id,
        &mut file_set,
    )?;
    write_pyproject_toml(env, mlx_noarch, &build.general, &mut file_set)?;
    write_setup_py(&mut file_set);
    write_metadata_for_backends(
        build,
        &build.general.backends,
        kernel_id,
        provenance,
        &mut file_set,
    )?;

    Ok(file_set)
}

fn write_ops_py(
    env: &Environment,
    name: &str,
    kernel_id: &KernelIdentifier,
    file_set: &mut FileSet,
) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("mlx-ext");
    path.push(name);
    path.push("_ops.py");
    let writer = file_set.entry(path);

    env.get_template("mlx/noarch/_ops.py")
        .wrap_err("Cannot get MLX noarch _ops.py template")?
        .render_captured_to(
            context! {
                kernel_name => kernel_id.name(),
                kernel_unique_id => kernel_id.unique_id(),
            },
            writer,
        )
        .wrap_err("Cannot render MLX noarch _ops.py template")?;

    Ok(())
}

fn write_pyproject_toml(
    env: &Environment,
    mlx_noarch: &MlxNoarch,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

    let data_globs = mlx_noarch
        .data_extensions()
        .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "));

    // Common python dependencies (no backend-specific ones)
    let python_dependencies =
        itertools::process_results(general.general_python_depends(), |iter| {
            iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
                .join(", ")
        })?;

    // Collect backend-specific dependencies for the kernel's backends.
    let mut backend_dependencies = Vec::new();
    for backend in &general.backends {
        let deps = itertools::process_results(general.backend_python_depends(*backend), |iter| {
            iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
                .join(", ")
        })?;

        if !deps.is_empty() {
            backend_dependencies.push((backend.to_string(), deps));
        }
    }

    env.get_template("mlx/noarch/pyproject.toml")
        .wrap_err("Cannot get MLX noarch pyproject.toml template")?
        .render_captured_to(
            context! {
                data_globs => data_globs,
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
                name => &general.name,
                python_name => general.name.python_name(),
            },
            writer,
        )
        .wrap_err("Cannot render MLX noarch pyproject.toml template")?;

    Ok(())
}

fn write_setup_py(file_set: &mut FileSet) {
    file_set
        .entry("setup.py")
        .extend_from_slice(SETUP_PY.as_bytes());
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use kernels_common::config::Build;

    use crate::pyproject::create_pyproject_file_set;
    use crate::pyproject::ops_identifier::KernelIdentifier;

    #[test]
    fn mlx_noarch_file_set() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("build.toml"),
            r#"
            [general]
            name = "mlx-demo"
            version = 1
            edition = 6
            license = "Apache-2.0"
            backends = ["metal"]

            [mlx-noarch]
            pyext = ["py", "json"]
            "#,
        )
        .unwrap();

        let build = Build::open(tmp.path()).unwrap();
        let kernel_id =
            KernelIdentifier::new(tmp.path(), "mlx_demo".to_owned(), Some("abc123".to_owned()));
        let file_set = create_pyproject_file_set(build, &kernel_id, None).unwrap();

        let mut names = file_set.into_names().into_iter().collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            [
                PathBuf::from("metadata-metal.json"),
                PathBuf::from("mlx-ext/mlx_demo/_ops.py"),
                PathBuf::from("pyproject.toml"),
                PathBuf::from("setup.py"),
            ]
        );
    }
}
