use std::io::Write;

use eyre::{Context, Result};
use itertools::Itertools;
use kernels_data::config::{Build, Kernel};
use minijinja::{context, Environment};

use crate::common::prefix_and_join_includes;

pub fn render_kernel_components(
    env: &Environment,
    build: &Build,
    write: &mut impl Write,
) -> Result<()> {
    for (kernel_name, kernel) in build.kernels.iter() {
        render_kernel_component(env, kernel_name, kernel, write)?;
    }

    Ok(())
}

fn render_kernel_component(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    write: &mut impl Write,
) -> Result<()> {
    // Easier to do in Rust than Jinja.
    let sources = kernel
        .src()
        .iter()
        .map(|src| format!("\"{src}\""))
        .collect_vec()
        .join("\n");

    match kernel {
        Kernel::Cpu { .. } => {
            render_kernel_component_cpu(env, kernel_name, kernel, sources, write)?
        }
        Kernel::Cuda { .. } => {
            render_kernel_component_cuda(env, kernel_name, kernel, sources, write)?
        }
        Kernel::Rocm { .. } => {
            render_kernel_component_hip(env, kernel_name, kernel, sources, write)?
        }
        Kernel::Metal { .. } => {
            render_kernel_component_metal(env, kernel_name, kernel, sources, write)?
        }
        Kernel::Xpu { .. } => {
            render_kernel_component_xpu(env, kernel_name, kernel, sources, write)?
        }
    }

    Ok(())
}

fn render_kernel_component_cpu(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/cpu.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_kernel_component_cuda(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    let (cuda_capabilities, cuda_flags, cuda_minver) = match kernel {
        Kernel::Cuda {
            cuda_capabilities,
            cuda_flags,
            cuda_minver,
            ..
        } => (
            cuda_capabilities.as_deref(),
            cuda_flags.as_deref(),
            cuda_minver.as_ref(),
        ),
        _ => unreachable!("Unsupported kernel type for CUDA rendering"),
    };

    env.get_template("kernel-component/cuda.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                name => kernel_name,
                cuda_capabilities => cuda_capabilities,
                cuda_flags => cuda_flags.map(|flags| flags.join(";")),
                cuda_minver => cuda_minver.map(ToString::to_string),
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_kernel_component_hip(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    let (rocm_archs, hip_flags) = match kernel {
        Kernel::Rocm {
            rocm_archs,
            hip_flags,
            ..
        } => (rocm_archs.as_deref(), hip_flags.as_deref()),
        _ => unreachable!("Unsupported kernel type for ROCm rendering"),
    };

    env.get_template("kernel-component/hip.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                rocm_archs => rocm_archs,
                hip_flags => hip_flags.map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_kernel_component_metal(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/metal.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_kernel_component_xpu(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    let sycl_flags = match kernel {
        Kernel::Xpu { sycl_flags, .. } => sycl_flags.as_deref(),
        _ => unreachable!("Unsupported kernel type for XPU rendering"),
    };

    env.get_template("kernel-component/xpu.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                sycl_flags => sycl_flags.map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}
