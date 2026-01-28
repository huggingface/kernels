use std::{io::Write, path::PathBuf};

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, Build, Torch};
use crate::fileset::FileSet;
use crate::torch::common::write_metadata;
use crate::torch::common::write_ops_py;
use crate::torch::common::write_pyproject_toml;
use crate::torch::common::write_torch_registration_macros;
use crate::torch::kernel::render_kernel_components;
use crate::torch::kernel_ops_identifier;
use crate::version::Version;

static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static COMPILE_METAL_CMAKE: &str = include_str!("../templates/metal/compile-metal.cmake");
static METALLIB_TO_HEADER_PY: &str = include_str!("../templates/metal/metallib_to_header.py");

pub fn write_torch_ext_metal(
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

    write_pyproject_toml(env, Backend::Metal, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(Backend::Metal, &build.general, &mut file_set)?;

    Ok(file_set)
}

fn write_cmake(
    env: &Environment,
    build: &Build,
    torch: &Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let mut utils_path = PathBuf::new();
    utils_path.push("cmake");
    utils_path.push("utils.cmake");
    file_set
        .entry(utils_path.clone())
        .extend_from_slice(CMAKE_UTILS.as_bytes());

    let mut kernel_path = PathBuf::new();
    kernel_path.push("cmake");
    kernel_path.push("kernel.cmake");
    file_set
        .entry(kernel_path.clone())
        .extend_from_slice(CMAKE_KERNEL.as_bytes());

    let mut compile_metal_path = PathBuf::new();
    compile_metal_path.push("cmake");
    compile_metal_path.push("compile-metal.cmake");
    file_set
        .entry(compile_metal_path)
        .extend_from_slice(COMPILE_METAL_CMAKE.as_bytes());

    let mut metallib_to_header_path = PathBuf::new();
    metallib_to_header_path.push("cmake");
    metallib_to_header_path.push("metallib_to_header.py");
    file_set
        .entry(metallib_to_header_path)
        .extend_from_slice(METALLIB_TO_HEADER_PY.as_bytes());

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(
        env,
        name,
        torch.minver.as_ref(),
        torch.maxver.as_ref(),
        cmake_writer,
    )?;

    // Add deps once we have any non-CUDA deps.
    // render_deps(env, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, name, ops_name, cmake_writer)?;

    Ok(())
}

fn render_binding(
    env: &Environment,
    torch: &Torch,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("metal/torch-binding.cmake")
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

pub fn render_extension(
    env: &Environment,
    name: &str,
    ops_name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("metal/torch-extension.cmake")
        .wrap_err("Cannot get Torch extension template")?
        .render_to_write(
            context! {
                name => name,
                ops_name => ops_name,
            },
            &mut *write,
        )
        .wrap_err("Cannot render Torch extension template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_preamble(
    env: &Environment,
    name: &str,
    torch_minver: Option<&Version>,
    torch_maxver: Option<&Version>,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("metal/preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => name,
                torch_minver => torch_minver.map(|v| v.to_string()),
                torch_maxver => torch_maxver.map(|v| v.to_string()),
            },
            &mut *write,
        )
        .wrap_err("Cannot render CMake prelude template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn write_setup_py(
    env: &Environment,
    torch: &Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch.data_globs().map(|globs| globs.join(", "));

    env.get_template("metal/setup.py")
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
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

fn prefix_and_join_includes<S>(includes: impl AsRef<[S]>) -> String
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
