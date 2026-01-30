use std::path::PathBuf;

use eyre::{bail, Result};
use minijinja::Environment;

use crate::config::{Backend, Build, Torch};
use crate::torch::common::{
    render_binding, render_extension, render_preamble, write_cmake_helpers, write_metadata,
    write_ops_py, write_pyproject_toml, write_setup_py, write_torch_registration_macros,
};
use crate::torch::deps::render_deps;
use crate::torch::kernel::render_kernel_components;
use crate::torch::kernel_ops_identifier;
use crate::FileSet;

pub fn write_torch_ext_cuda(
    env: &Environment,
    backend: Backend,
    build: &Build,
    target_dir: PathBuf,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let torch_ext = match build.torch.as_ref() {
        Some(torch_ext) => torch_ext,
        None => bail!("Build configuration does not have `torch` section"),
    };

    let mut file_set = FileSet::default();

    let ops_name = kernel_ops_identifier(&target_dir, &build.general.python_name(), ops_id);

    write_cmake(
        env,
        backend,
        build,
        torch_ext,
        &build.general.name,
        &ops_name,
        &mut file_set,
    )?;

    write_setup_py(
        env,
        torch_ext,
        &build.general.name,
        &ops_name,
        &mut file_set,
    )?;

    write_ops_py(env, &build.general.python_name(), &ops_name, &mut file_set)?;

    write_pyproject_toml(env, backend, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(backend, &build.general, &mut file_set)?;

    Ok(file_set)
}

fn write_cmake(
    env: &Environment,
    backend: Backend,
    build: &Build,
    torch: &Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    write_cmake_helpers(file_set);

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(
        env,
        name,
        build.general.cuda.as_ref().and_then(|c| c.minver.as_ref()),
        build.general.cuda.as_ref().and_then(|c| c.maxver.as_ref()),
        torch.minver.as_ref(),
        torch.maxver.as_ref(),
        cmake_writer,
    )?;

    render_deps(env, backend, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, name, ops_name, cmake_writer)?;

    Ok(())
}
