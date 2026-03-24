use std::io::Write;
use std::path::{Path, PathBuf};

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::context;

use crate::pyproject::common::{
    prefix_and_join_includes, write_cmake_file, write_compat_py, write_metadata,
};
use crate::pyproject::ops_identifier::{git_identifier, random_identifier};
use crate::pyproject::FileSet;
use kernels_data::config::{Backend, Build, General, Torch};

use crate::pyproject::deps::render_deps;

use crate::pyproject::kernel::render_kernel_components;

mod noarch;
pub use noarch::write_torch_ext_noarch;

static BUILD_VARIANTS_UTILS: &str = include_str!("../templates/torch/build-variants.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static COMPILE_METAL_CMAKE: &str = include_str!("../templates/torch/metal/compile-metal.cmake");
static GET_GPU_LANG: &str = include_str!("../templates/torch/get_gpu_lang.cmake");
static GET_GPU_LANG_PY: &str = include_str!("../templates/torch/get_gpu_lang.py");
static ADD_GPU_ARCH_METADATA_PY: &str = include_str!("../templates/torch/add_gpu_arch_metadata.py");
static HIPIFY: &str = include_str!("../templates/torch/cuda/hipify.py");
static METALLIB_TO_HEADER_PY: &str = include_str!("../templates/torch/metal/metallib_to_header.py");
static REGISTRATION_H: &str = include_str!("../templates/torch/registration.h");
static OPS_PY_IN: &str = include_str!("../templates/torch/_ops.py.in");

fn write_setup_py(
    env: &minijinja::Environment,
    general: &General,
    torch: &Torch,
    revision: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch
        .data_extensions()
        .map(|exts| exts.iter().map(|ext| format!("\"**/*.{ext}\"")).join(", "));

    env.get_template("torch/setup.py")
        .wrap_err("Cannot get setup.py template")?
        .render_to_write(
            context! {
                data_globs => data_globs,
                revision => revision,
                python_name => general.name.python_name(),
                version => "0.1.0",
            },
            writer,
        )
        .wrap_err("Cannot render setup.py template")?;

    Ok(())
}

fn write_pyproject_toml(
    env: &minijinja::Environment,
    general: &General,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("pyproject.toml");

    // Common python dependencies (no backend-specific ones)
    let python_dependencies = itertools::process_results(general.python_depends(), |iter| {
        iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
            .join(", ")
    })?;

    // Collect backend-specific dependencies for all backends
    let mut backend_dependencies = Vec::new();
    for backend in &Backend::all() {
        let deps = itertools::process_results(general.backend_python_depends(*backend), |iter| {
            iter.flat_map(|(_, deps)| deps.python.iter().map(|d| format!("\"{}\"", d.pkg)))
                .join(", ")
        })?;

        if !deps.is_empty() {
            backend_dependencies.push((backend.to_string(), deps));
        }
    }

    env.get_template("torch/pyproject.toml")
        .wrap_err("Cannot get pyproject.toml template")?
        .render_to_write(
            context! {
                python_name => general.name.python_name(),
                python_dependencies => python_dependencies,
                backend_dependencies => backend_dependencies,
            },
            writer,
        )
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

fn write_torch_registration_macros(file_set: &mut FileSet) -> Result<()> {
    let mut path = PathBuf::new();
    path.push("torch-ext");
    path.push("registration.h");
    file_set
        .entry(path)
        .extend_from_slice(REGISTRATION_H.as_bytes());

    Ok(())
}

fn render_binding(
    env: &minijinja::Environment,
    torch: &Torch,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("torch/torch-binding.cmake")
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

/// Writes all CMake helper files that any backend might need.
/// Each backend will use only the files it references in its CMakeLists.txt.
fn write_cmake_helpers(file_set: &mut FileSet) {
    write_cmake_file(file_set, "utils.cmake", CMAKE_UTILS.as_bytes());
    write_cmake_file(file_set, "kernel.cmake", CMAKE_KERNEL.as_bytes());
    write_cmake_file(
        file_set,
        "build-variants.cmake",
        BUILD_VARIANTS_UTILS.as_bytes(),
    );
    write_cmake_file(
        file_set,
        "add_gpu_arch_metadata.py",
        ADD_GPU_ARCH_METADATA_PY.as_bytes(),
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

fn render_extension(
    env: &minijinja::Environment,
    general: &General,
    torch: &Torch,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("torch/torch-extension.cmake")
        .wrap_err("Cannot get Torch extension template")?
        .render_to_write(
            context! {
                python_name => general.name.python_name(),
                data_extensions => torch.data_extensions(),
            },
            &mut *write,
        )
        .wrap_err("Cannot render Torch extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_preamble(
    env: &minijinja::Environment,
    general: &General,
    torch: &Torch,
    revision: &str,
    write: &mut impl Write,
) -> Result<()> {
    let cuda_minver = general.cuda.as_ref().and_then(|c| c.minver.as_ref());
    let cuda_maxver = general.cuda.as_ref().and_then(|c| c.maxver.as_ref());

    env.get_template("torch/preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => general.name.as_str(),
                python_name => general.name.python_name(),
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

fn write_cmake(
    env: &minijinja::Environment,
    build: &Build,
    torch: &Torch,
    name: &str,
    revision: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    write_cmake_helpers(file_set);

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(env, &build.general, torch, revision, cmake_writer)?;

    render_deps(env, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, &build.general, torch, cmake_writer)?;

    Ok(())
}

pub fn write_torch_ext(
    env: &minijinja::Environment,
    build: &Build,
    target_dir: impl AsRef<Path>,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let torch_ext = match build.framework.torch() {
        Some(torch_ext) => torch_ext,
        None => bail!("Build configuration does not have `torch` section"),
    };

    let mut file_set = FileSet::default();

    let revision = ops_id
        .unwrap_or_else(|| git_identifier(&target_dir).unwrap_or_else(|_| random_identifier()));

    write_cmake(
        env,
        build,
        torch_ext,
        build.general.name.as_str(),
        &revision,
        &mut file_set,
    )?;

    write_setup_py(env, &build.general, torch_ext, &revision, &mut file_set)?;

    write_compat_py(&mut file_set)?;

    write_pyproject_toml(env, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(&build.general, &mut file_set)?;

    Ok(file_set)
}
