use std::io::Write;
use std::path::PathBuf;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, General, Torch};
use crate::metadata::Metadata;
use crate::FileSet;

static REGISTRATION_H: &str = include_str!("../templates/registration.h");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static WINDOWS_UTILS: &str = include_str!("../templates/windows.cmake");
static HIPIFY: &str = include_str!("../templates/cuda/hipify.py");
static COMPILE_METAL_CMAKE: &str = include_str!("../templates/metal/compile-metal.cmake");
static METALLIB_TO_HEADER_PY: &str = include_str!("../templates/metal/metallib_to_header.py");
static GET_GPU_LANG: &str = include_str!("../templates/get_gpu_lang.cmake");
static GET_GPU_LANG_PY: &str = include_str!("../templates/get_gpu_lang.py");

pub fn write_setup_py(
    env: &Environment,
    torch: &crate::config::Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch.data_globs().map(|globs| globs.join(", "));

    env.get_template("setup.py")
        .wrap_err("Cannot get setup.py template")?
        .render_to_write(
            context! {
                data_globs => data_globs,
                ops_name => ops_name,
                name => name,
                version => "0.1.0",
            },
            writer,
        )
        .wrap_err("Cannot render setup.py template")?;

    Ok(())
}

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
    write_cmake_file(file_set, "get_gpu_lang.cmake", GET_GPU_LANG.as_bytes());
    write_cmake_file(file_set, "get_gpu_lang.py", GET_GPU_LANG_PY.as_bytes());
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
