use std::path::PathBuf;

use eyre::Result;
use itertools::Itertools;

use kernels_data::config::{Backend, Build};
use kernels_data::metadata::{BuildInfo, KernelBuilderInfo, Metadata};

use crate::pyproject::ops_identifier::KernelIdentifier;
use crate::pyproject::FileSet;

static COMPAT_PY: &str = include_str!("templates/compat.py");
static ADD_BUILD_METADATA_PY: &str = include_str!("templates/torch/add_build_metadata.py");

pub fn write_compat_py(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("compat.py");
    file_set.entry(path).extend_from_slice(COMPAT_PY.as_bytes());

    Ok(())
}

/// `kernel-builder` provenance baked in at compile time.
///
/// The git provenance is gathered by the `built` crate (see `build.rs`). In
/// build sandboxes without a `.git` (e.g. Nix), the derivation supplies it via
/// `built`'s `BUILT_OVERRIDE_hf_kernel_builder_GIT_*` environment variables.
pub(crate) fn kernel_builder_info() -> KernelBuilderInfo {
    KernelBuilderInfo {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        sha: crate::built_info::GIT_COMMIT_HASH.map(str::to_owned),
        dirty: crate::built_info::GIT_DIRTY.unwrap_or(false),
    }
}

pub fn write_metadata(
    build: &Build,
    kernel_id: &KernelIdentifier,
    build_info: Option<&BuildInfo>,
    file_set: &mut FileSet,
) -> Result<()> {
    for backend in &Backend::all() {
        let writer = file_set.entry(format!("metadata-{backend}.json"));

        let mut metadata =
            Metadata::for_backend(build, kernel_id.to_string_for_backend(*backend), *backend)?;
        metadata.build_info = build_info.cloned();

        serde_json::to_writer_pretty(writer, &metadata)?;
    }

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

pub fn write_add_build_metadata_py(file_set: &mut FileSet) {
    write_cmake_file(
        file_set,
        "add_build_metadata.py",
        ADD_BUILD_METADATA_PY.as_bytes(),
    );
}

/// Helper function to write a file to the cmake subdirectory
pub fn write_cmake_file(file_set: &mut FileSet, filename: &str, content: &[u8]) {
    let mut path = PathBuf::new();
    path.push("cmake");
    path.push(filename);
    file_set.entry(path).extend_from_slice(content);
}
