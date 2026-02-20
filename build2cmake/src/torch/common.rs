use std::io::Write;
use std::path::{Path, PathBuf};

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, Build, General, Torch};
use crate::metadata::Metadata;
use crate::ops_identifier::{git_identifier, random_identifier};
use crate::torch::deps::render_deps;
use crate::torch::kernel::render_kernel_components;
use crate::FileSet;

static BUILD_VARIANTS_UTILS: &str = include_str!("../templates/build-variants.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static COMPAT_PY: &str = include_str!("../templates/compat.py");
static COMPILE_METAL_CMAKE: &str = include_str!("../templates/metal/compile-metal.cmake");
static GET_GPU_LANG: &str = include_str!("../templates/get_gpu_lang.cmake");
static GET_GPU_LANG_PY: &str = include_str!("../templates/get_gpu_lang.py");
static HIPIFY: &str = include_str!("../templates/cuda/hipify.py");
static METALLIB_TO_HEADER_PY: &str = include_str!("../templates/metal/metallib_to_header.py");
static REGISTRATION_H: &str = include_str!("../templates/registration.h");
static OPS_PY_IN: &str = include_str!("../templates/_ops.py.in");

pub fn write_setup_py(
    env: &Environment,
    general: &General,
    torch: &crate::config::Torch,
    revision: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch
        .data_extensions()
        .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "));

    env.get_template("setup.py")
        .wrap_err("Cannot get setup.py template")?
        .render_to_write(
            context! {
                data_globs => data_globs,
                revision => revision,
                python_name => general.python_name(),
                version => "0.1.0",
            },
            writer,
        )
        .wrap_err("Cannot render setup.py template")?;

    Ok(())
}

pub fn write_compat_py(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("compat.py");
    file_set.entry(path).extend_from_slice(COMPAT_PY.as_bytes());

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
        let writer = file_set.entry(format!("metadata-{}.json", backend));

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
    write_cmake_file(
        file_set,
        "build-variants.cmake",
        BUILD_VARIANTS_UTILS.as_bytes(),
    );
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
    write_cmake_file(file_set, "_ops.py.in", OPS_PY_IN.as_bytes());
}

pub fn render_extension(
    env: &Environment,
    general: &General,
    torch: &Torch,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("torch-extension.cmake")
        .wrap_err("Cannot get Torch extension template")?
        .render_to_write(
            context! {
                python_name => general.python_name(),
                data_extensions => torch.data_extensions(),
            },
            &mut *write,
        )
        .wrap_err("Cannot render Torch extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}

pub fn render_preamble(
    env: &Environment,
    general: &General,
    torch: &Torch,
    target_dir: impl AsRef<Path>,
    write: &mut impl Write,
) -> Result<()> {
    let cuda_minver = general.cuda.as_ref().and_then(|c| c.minver.as_ref());
    let cuda_maxver = general.cuda.as_ref().and_then(|c| c.maxver.as_ref());
    let revision = git_identifier(&target_dir).unwrap_or_else(|_| random_identifier());

    env.get_template("preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => &general.name,
                python_name => general.python_name(),
                revision => revision,
                cuda_minver => cuda_minver.map(|v| v.to_string()),
                cuda_maxver => cuda_maxver.map(|v| v.to_string()),
                torch_minver => torch.minver.as_ref().map(|v| v.to_string()),
                torch_maxver => torch.maxver.as_ref().map(|v| v.to_string()),
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
    target_dir: impl AsRef<Path>,
    torch: &Torch,
    name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    write_cmake_helpers(file_set);

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(env, &build.general, torch, &target_dir, cmake_writer)?;

    render_deps(env, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, &build.general, torch, cmake_writer)?;

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

    let revision = ops_id
        .unwrap_or_else(|| git_identifier(&target_dir).unwrap_or_else(|_| random_identifier()));

    write_cmake(
        env,
        build,
        &target_dir,
        torch_ext,
        &build.general.name,
        &mut file_set,
    )?;

    write_setup_py(env, &build.general, torch_ext, &revision, &mut file_set)?;

    write_compat_py(&mut file_set)?;

    write_pyproject_toml(env, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(&build.general, &mut file_set)?;

    Ok(file_set)
}
