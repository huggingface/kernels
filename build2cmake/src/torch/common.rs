use std::io::Write;
use std::path::PathBuf;

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, Build, General, Torch};
use crate::metadata::Metadata;
use crate::torch::deps::render_deps;
use crate::torch::kernel::render_kernel_components;
use crate::version::Version;
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
    general: &General,
    torch: &crate::config::Torch,
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
                python_name => general.python_name(),
                version => "0.1.0",
            },
            writer,
        )
        .wrap_err("Cannot render setup.py template")?;

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

    env.get_template("pyproject.toml")
        .wrap_err("Cannot get pyproject.toml template")?
        .render_to_write(
            context! {
                python_name => general.python_name(),
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

pub fn write_metadata(general: &General, file_set: &mut FileSet) -> Result<()> {
    for backend in &Backend::all() {
        let writer = file_set.entry(format!("metadata-{}.json", backend.to_string()));

        let python_depends = general
            .python_depends()
            .chain(general.backend_python_depends(*backend))
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

pub fn render_preamble(
    env: &Environment,
    name: &str,
    cuda_minver: Option<&Version>,
    cuda_maxver: Option<&Version>,
    torch_minver: Option<&Version>,
    torch_maxver: Option<&Version>,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => name,
                cuda_minver => cuda_minver.map(|v| v.to_string()),
                cuda_maxver => cuda_maxver.map(|v| v.to_string()),
                torch_minver => torch_minver.map(|v| v.to_string()),
                torch_maxver => torch_maxver.map(|v| v.to_string()),
                platform => std::env::consts::OS
            },
            &mut *write,
        )
        .wrap_err("Cannot render CMake prelude template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn write_cmake(
    env: &Environment,
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

    render_deps(env, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, name, ops_name, cmake_writer)?;

    Ok(())
}

pub fn write_torch_ext(
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

    let ops_name = crate::torch::ops_identifier::kernel_ops_identifier(
        &target_dir,
        &build.general.python_name(),
        ops_id,
    );

    write_cmake(
        env,
        build,
        torch_ext,
        &build.general.name,
        &ops_name,
        &mut file_set,
    )?;

    write_setup_py(env, &build.general, torch_ext, &ops_name, &mut file_set)?;

    write_ops_py(env, &build.general.python_name(), &ops_name, &mut file_set)?;

    write_pyproject_toml(env, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(&build.general, &mut file_set)?;

    Ok(file_set)
}
