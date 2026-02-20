use std::io::Write;
use std::path::{Path, PathBuf};

use eyre::{bail, Context, Result};
use minijinja::{context, Environment};

use crate::config::{Backend, Build, General, TvmFfi};
use crate::ops_identifier::{git_identifier, random_identifier};
use crate::torch::common::write_cmake_file;
use crate::torch::kernel::render_kernel_components;
use crate::FileSet;

static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");

fn write_cmake_helpers(file_set: &mut FileSet) {
    write_cmake_file(file_set, "utils.cmake", CMAKE_UTILS.as_bytes());
    write_cmake_file(file_set, "kernel.cmake", CMAKE_KERNEL.as_bytes());
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

    write_cmake(
        env,
        build,
        &target_dir,
        tvm_ffi_ext,
        &build.general.name,
        &mut file_set,
    )?;

    Ok(file_set)
}

pub fn render_extension(
    env: &Environment,
    general: &General,
    tvm_ffi: &TvmFfi,
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

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, &build.general, tvm_ffi, cmake_writer)?;

    Ok(())
}
