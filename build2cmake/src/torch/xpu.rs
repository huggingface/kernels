use std::path::PathBuf;

use eyre::{bail, Result};
use minijinja::Environment;

use crate::config::{Backend, Build};
use crate::torch::common::{
    write_cmake, write_metadata, write_ops_py, write_pyproject_toml, write_setup_py,
    write_torch_registration_macros,
};
use crate::torch::kernel_ops_identifier;
use crate::FileSet;

pub fn write_torch_ext_xpu(
    env: &Environment,
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
        Backend::Xpu,
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

    write_pyproject_toml(env, Backend::Xpu, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(Backend::Xpu, &build.general, &mut file_set)?;

    Ok(file_set)
}
