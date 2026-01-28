use std::collections::HashSet;
use std::env;
use std::io::Write;
use std::path::PathBuf;

use eyre::{bail, Context, Result};
use minijinja::{context, Environment};

use crate::config::{Backend, Build, Dependency, Torch};
use crate::torch::common::prefix_and_join_includes;
use crate::torch::common::write_metadata;
use crate::torch::common::write_pyproject_toml;
use crate::torch::kernel::render_kernel_components;
use crate::torch::kernel_ops_identifier;
use crate::version::Version;
use crate::FileSet;

static CMAKE_UTILS: &str = include_str!("../templates/utils.cmake");
static CMAKE_KERNEL: &str = include_str!("../templates/kernel.cmake");
static WINDOWS_UTILS: &str = include_str!("../templates/windows.cmake");
static REGISTRATION_H: &str = include_str!("../templates/registration.h");
static HIPIFY: &str = include_str!("../templates/cuda/hipify.py");

pub fn write_torch_ext_cuda(
    env: &Environment,
    backend: Backend,
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
        backend,
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

    write_pyproject_toml(env, backend, &build.general, &mut file_set)?;

    write_torch_registration_macros(&mut file_set)?;

    write_metadata(backend, &build.general, &mut file_set)?;

    Ok(file_set)
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

fn write_setup_py(
    env: &Environment,
    torch: &Torch,
    name: &str,
    ops_name: &str,
    file_set: &mut FileSet,
) -> Result<()> {
    let writer = file_set.entry("setup.py");

    let data_globs = torch.data_globs().map(|globs| globs.join(", "));

    env.get_template("cuda/setup.py")
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
        .wrap_err("Cannot render kernel template")?;

    Ok(())
}

fn write_cmake(
    env: &Environment,
    backend: Backend,
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

    let mut windows_utils_path = PathBuf::new();
    windows_utils_path.push("cmake");
    windows_utils_path.push("windows.cmake");
    file_set
        .entry(windows_utils_path.clone())
        .extend_from_slice(WINDOWS_UTILS.as_bytes());

    let mut hipify_path = PathBuf::new();
    hipify_path.push("cmake");
    hipify_path.push("hipify.py");
    file_set
        .entry(hipify_path.clone())
        .extend_from_slice(HIPIFY.as_bytes());

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

    render_deps(env, backend, build, cmake_writer)?;

    render_binding(env, torch, name, cmake_writer)?;

    render_kernel_components(env, build, cmake_writer)?;

    render_extension(env, name, ops_name, cmake_writer)?;

    Ok(())
}

pub fn render_binding(
    env: &Environment,
    torch: &Torch,
    name: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("cuda/torch-binding.cmake")
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
            Dependency::Cutlass2_10 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "2.10.0",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass3_5 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "3.5.1",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass3_6 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "3.6.0",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass3_8 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "3.8.0",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass3_9 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "3.9.2",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass4_0 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_to_write(
                        context! {
                            version => "4.0.0",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Torch => (),
            _ => {
                eprintln!("Warning: CUDA backend doesn't need/support dependency: {dep:?}");
            }
        };
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
    env.get_template("cuda/torch-extension.cmake")
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
    env.get_template("cuda/preamble.cmake")
        .wrap_err("Cannot get CMake prelude template")?
        .render_to_write(
            context! {
                name => name,
                cuda_minver => cuda_minver.map(|v| v.to_string()),
                cuda_maxver => cuda_maxver.map(|v| v.to_string()),
                torch_minver => torch_minver.map(|v| v.to_string()),
                torch_maxver => torch_maxver.map(|v| v.to_string()),
                platform => env::consts::OS
            },
            &mut *write,
        )
        .wrap_err("Cannot render CMake prelude template")?;

    write.write_all(b"\n")?;

    Ok(())
}
