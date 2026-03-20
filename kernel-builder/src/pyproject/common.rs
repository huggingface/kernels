use std::path::PathBuf;

use eyre::Result;
use itertools::Itertools;

use crate::config::{Backend, General};
use crate::pyproject::metadata::Metadata;
use crate::pyproject::FileSet;

static COMPAT_PY: &str = include_str!("templates/compat.py");
static COMPILE_METAL_CMAKE: &str = include_str!("templates/torch/metal/compile-metal.cmake");
static METALLIB_TO_HEADER_PY: &str = include_str!("templates/torch/metal/metallib_to_header.py");

pub fn write_compat_py(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("compat.py");
    file_set.entry(path).extend_from_slice(COMPAT_PY.as_bytes());

    Ok(())
}

pub fn write_metadata(general: &General, file_set: &mut FileSet) -> Result<()> {
    for backend in &Backend::all() {
        let writer = file_set.entry(format!("metadata-{}.json", backend));

        let python_depends = general
            .python_depends()
            .map(|deps| Ok(deps?.0.to_owned()))
            .chain(
                general
                    .backend_python_depends(*backend)
                    .map(|deps| Ok(deps?.0.to_owned())),
            )
            .collect::<Result<Vec<_>>>()?;

        let metadata = Metadata {
            version: general.version,
            license: general.license.clone(),
            python_depends,
        };

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

/// Helper function to write a file to the cmake subdirectory
pub fn write_cmake_file(file_set: &mut FileSet, filename: &str, content: &[u8]) {
    let mut path = PathBuf::new();
    path.push("cmake");
    path.push(filename);
    file_set.entry(path).extend_from_slice(content);
}

pub fn write_compile_metal_cmake(file_set: &mut FileSet) {
    write_cmake_file(file_set, "compile-metal.cmake", COMPILE_METAL_CMAKE.as_bytes());
}

pub fn write_metallib_to_header_py(file_set: &mut FileSet) {
    write_cmake_file(file_set, "metallib_to_header.py", METALLIB_TO_HEADER_PY.as_bytes());
}
