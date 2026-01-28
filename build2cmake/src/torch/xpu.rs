use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;

use eyre::{bail, Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Backend, Build, Dependency, Torch};
use crate::torch::common::write_metadata;
use crate::torch::common::write_pyproject_toml;
use crate::torch::common::write_torch_registration_macros;
use crate::torch::kernel::render_kernel_components;
use crate::torch::kernel_ops_identifier;
use crate::version::Version;
use crate::FileSet;

static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static WINDOWS_UTILS: &str = include_str!("../templates/windows.cmake");

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

fn write_setup_py(
    env: &Environment,
    torch: &Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch.data_globs().map(|globs| globs.join(", "));

    env.get_template("xpu/setup.py")
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

fn write_ops_py(
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
        .wrap_err("Cannot render _ops.py template")?;

    Ok(())
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

    // Add windows.cmake for Windows-specific install targets
    let mut windows_utils_path = PathBuf::new();
    windows_utils_path.push("cmake");
    windows_utils_path.push("windows.cmake");
    file_set
        .entry(windows_utils_path.clone())
        .extend_from_slice(WINDOWS_UTILS.as_bytes());

    let mut kernel_path = PathBuf::new();
    kernel_path.push("cmake");
    kernel_path.push("kernel.cmake");
    file_set
        .entry(kernel_path.clone())
        .extend_from_slice(CMAKE_KERNEL.as_bytes());

    let cmake_writer = file_set.entry("CMakeLists.txt");

    render_preamble(
        env,
        name,
        torch.minver.as_ref(),
        torch.maxver.as_ref(),
        cmake_writer,
    )?;

    render_deps(env, Backend::Xpu, build, cmake_writer)?;

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
    env.get_template("xpu/torch-binding.cmake")
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

fn render_deps(
    env: &Environment,
    backend: Backend,
    build: &Build,
    write: &mut impl Write,
) -> Result<()> {
    let mut deps = HashSet::new();

    for kernel in build
        .kernels
        .values()
        .filter(|kernel| kernel.backend() == backend)
    {
        deps.extend(kernel.depends());
    }

    for dep in deps {
        match dep {
            Dependency::CutlassSycl => {
                env.get_template("xpu/dep-cutlass-sycl.cmake")?
                    .render_to_write(context! {}, &mut *write)?;
            }
            Dependency::Torch => (),
            _ => {
                // XPU supports CUTLASS-SYCL instead of CUTLASS
                eprintln!("Warning: XPU backend doesn't need/support dependency: {dep:?}");
            }
        }
        write.write_all(b"\n")?;
    }

    Ok(())
}

pub fn render_extension(
    env: &Environment,
    name: &str,
    ops_name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("xpu/torch-extension.cmake")
        .wrap_err("Cannot get Torch extension template")?
        .render_to_write(
            context! {
                name => name,
                ops_name => ops_name,
                platform => std::env::consts::OS
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
    torch_minver: Option<&Version>,
    torch_maxver: Option<&Version>,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("xpu/preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => name,
                torch_minver => torch_minver.map(|v| v.to_string()),
                torch_maxver => torch_maxver.map(|v| v.to_string()),
                platform => std::env::consts::OS,
            },
            &mut *write,
        )
        .wrap_err("Cannot render CMake prelude template")?;

    write.write_all(b"\n")?;

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
