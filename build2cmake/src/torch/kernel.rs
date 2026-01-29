use std::io::Write;

use eyre::{Context, Result};
use itertools::Itertools;
use minijinja::{context, Environment};

use crate::config::{Build, Kernel};
use crate::torch::common::prefix_and_join_includes;

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
        Kernel::Cuda { .. } | Kernel::Rocm { .. } => {
            render_kernel_component_cuda(env, kernel_name, kernel, sources, write)?
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
    env.get_template("cpu/kernel.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_to_write(
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
    let (cuda_capabilities, rocm_archs, cuda_flags, hip_flags, cuda_minver) = match kernel {
        Kernel::Cuda {
            cuda_capabilities,
            cuda_flags,
            cuda_minver,
            ..
        } => (
            cuda_capabilities.as_deref(),
            None,
            cuda_flags.as_deref(),
            None,
            cuda_minver.as_ref(),
        ),
        Kernel::Rocm {
            rocm_archs,
            hip_flags,
            ..
        } => (
            None,
            rocm_archs.as_deref(),
            None,
            hip_flags.as_deref(),
            None,
        ),
        _ => unreachable!("Unsupported kernel type for CUDA rendering"),
    };

    env.get_template("cuda/kernel.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_to_write(
            context! {
                cuda_capabilities => cuda_capabilities,
                cuda_flags => cuda_flags.map(|flags| flags.join(";")),
                cuda_minver => cuda_minver.map(ToString::to_string),
                cxx_flags => kernel.cxx_flags().map(|flags| flags.join(";")),
                rocm_archs => rocm_archs,
                hip_flags => hip_flags.map(|flags| flags.join(";")),
                includes => kernel.include().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                supports_hipify => matches!(kernel, Kernel::Rocm{ .. }),
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
    env.get_template("metal/kernel.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_to_write(
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

    env.get_template("xpu/kernel.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_to_write(
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
