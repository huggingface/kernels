use std::path::Path;

use eyre::Result;
use minijinja::Environment;

mod common;
pub mod fileset;
mod kernel;
mod metadata;
mod ops_identifier;
mod torch;
mod tvm_ffi;

pub use fileset::FileSet;

pub fn create_pyproject_file_set(
    build: crate::config::Build,
    target_dir: impl AsRef<Path>,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    minijinja_embed::load_templates!(&mut env);

    let file_set = if matches!(build.framework, crate::config::Framework::TvmFfi(_)) {
        tvm_ffi::write_tvm_ffi_ext(&env, &build, target_dir, ops_id)?
    } else if build.is_noarch() {
        torch::write_torch_ext_noarch(&env, &build, target_dir, ops_id)?
    } else {
        torch::write_torch_ext(&env, &build, target_dir, ops_id)?
    };

    Ok(file_set)
}
