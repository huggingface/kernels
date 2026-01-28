use std::path::PathBuf;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, General};
use crate::metadata::Metadata;
use crate::FileSet;

static REGISTRATION_H: &str = include_str!("../templates/registration.h");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static WINDOWS_UTILS: &str = include_str!("../templates/windows.cmake");
static HIPIFY: &str = include_str!("../templates/cuda/hipify.py");
static COMPILE_METAL_CMAKE: &str = include_str!("../templates/metal/compile-metal.cmake");
static METALLIB_TO_HEADER_PY: &str = include_str!("../templates/metal/metallib_to_header.py");

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

pub fn write_ops_py(
    env: &Environment,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("torch-ext");
    path.push(name);
    path.push("_ops.py");
    let writer = file_set.entry(path);

    env.get_template("_ops.py")
        .wrap_err("Cannot get _ops.py template")?
        .render_to_write(
            context! {
                ops_name => ops_name,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

/// Helper function to write a file to the cmake subdirectory
pub fn write_cmake_file(file_set: &mut FileSet, filename: &str, content: &[u8]) {
    let mut path = PathBuf::new();
    path.push("cmake");
    path.push(filename);
    file_set.entry(path).extend_from_slice(content);
}

/// Writes all CMake helper files that any backend might need.
/// Each backend will use only the files it references in its CMakeLists.txt.
pub fn write_cmake_helpers(file_set: &mut FileSet) {
    write_cmake_file(file_set, "utils.cmake", CMAKE_UTILS.as_bytes());
    write_cmake_file(file_set, "kernel.cmake", CMAKE_KERNEL.as_bytes());
    write_cmake_file(file_set, "windows.cmake", WINDOWS_UTILS.as_bytes());
    write_cmake_file(file_set, "hipify.py", HIPIFY.as_bytes());
    write_cmake_file(
        file_set,
        "compile-metal.cmake",
        COMPILE_METAL_CMAKE.as_bytes(),
    );
    write_cmake_file(
        file_set,
        "metallib_to_header.py",
        METALLIB_TO_HEADER_PY.as_bytes(),
    );
}
