use std::io::Write;
use std::path::{Path, PathBuf};

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, Build, General, TvmFfi};
use crate::ops_identifier::{git_identifier, random_identifier};
use crate::torch::common::{prefix_and_join_includes, write_cmake_file};
use crate::torch::kernel::render_kernel_components;
use crate::FileSet;

static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static OPS_PY_IN: &str = include_str!("../templates/tvm_ffi/_ops.py.in");

fn write_cmake_helpers(file_set: &mut FileSet) {
    write_cmake_file(file_set, "utils.cmake", CMAKE_UTILS.as_bytes());
    write_cmake_file(file_set, "kernel.cmake", CMAKE_KERNEL.as_bytes());
    write_cmake_file(file_set, "_ops.py.in", OPS_PY_IN.as_bytes());
}

pub fn write_tvm_ffi_ext(
    env: &Environment,
    build: &Build,
    target_dir: PathBuf,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let tvm_ffi_ext = match build.tvm_ffi.as_ref() {
        Some(torch_ext) => torch_ext,
        None => bail!("Build configuration does not have `tvm-ffi` section"),
    };

    let mut file_set = FileSet::default();

    let revision = ops_id
        .unwrap_or_else(|| git_identifier(&target_dir).unwrap_or_else(|_| random_identifier()));

    write_cmake(
        env,
        build,
        &target_dir,
        tvm_ffi_ext,
        &build.general.name,
        &mut file_set,
    )?;

    write_setup_py(env, &build.general, tvm_ffi_ext, &revision, &mut file_set)?;

    write_pyproject_toml(env, &build.general, &mut file_set)?;

    Ok(file_set)
}

pub fn write_setup_py(
    env: &Environment,
    general: &General,
    tvm_ffi: &TvmFfi,
    revision: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = tvm_ffi
        .data_extensions()
        .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "));

    env.get_template("tvm_ffi/setup.py")
        .wrap_err("Cannot get tvm_ffi setup.py template")?
        .render_to_write(
            context! {
                data_globs => data_globs,
                revision => revision,
                python_name => general.python_name(),
            },
            writer,
        )
        .wrap_err("Cannot render tvm_ffi setup.py template")?;

    Ok(())
}

pub fn write_pyproject_toml(
    env: &Environment,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

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

    env.get_template("tvm_ffi/pyproject.toml")
        .wrap_err("Cannot get tvm_ffi pyproject.toml template")?
        .render_to_write(
            context! {
                python_name => general.python_name(),
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
            },
            writer,
        )
        .wrap_err("Cannot render tvm_ffi pyproject.toml template")?;

    Ok(())
}

pub fn render_binding(
    env: &Environment,
    tvm_ffi: &TvmFfi,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("tvm_ffi/binding.cmake")
        .wrap_err("Cannot get tvm_ffi binding template")?
        .render_to_write(
            context! {
                includes => tvm_ffi.include.as_ref().map(prefix_and_join_includes),
                name => name,
                src => tvm_ffi.src,
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi binding template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_extension(
    env: &Environment,
    general: &General,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("tvm_ffi/tvm-ffi-extension.cmake")
        .wrap_err("Cannot get tvm_ffi extension template")?
        .render_to_write(
            context! {
                python_name => general.python_name(),
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_preamble(
    env: &Environment,
    general: &General,
    target_dir: impl AsRef<Path>,
    write: &mut impl Write,
) -> Result<()> {
    let revision = git_identifier(&target_dir).unwrap_or_else(|_| random_identifier());

    env.get_template("tvm_ffi/preamble.cmake")
        .wrap_err("Cannot get tvm_ffi preamble template")?
        .render_to_write(
            context! {
                name => &general.name,
                python_name => general.python_name(),
                revision => revision,
            },
            &mut *write,
        )
        .wrap_err("Cannot render tvm_ffi preamble template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn write_cmake(
    env: &Environment,
    build: &Build,
    target_dir: impl AsRef<Path>,
    tvm_ffi: &TvmFfi,
    name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    write_cmake_helpers(file_set);

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(env, &build.general, &target_dir, cmake_writer)?;

    render_binding(env, tvm_ffi, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, &build.general, cmake_writer)?;

    Ok(())
}
