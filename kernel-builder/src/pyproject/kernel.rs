use std::io::Write;

use eyre::{Context, Result};
use itertools::Itertools;
use kernels_common::config::{Build, CppCpu, CppCuda, CppMetal, CppRocm, CppXpu, Kernel};
use minijinja::{context, Environment};

use crate::pyproject::common::prefix_and_join_includes;

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
        Kernel::CppCpu(cpu) => render_kernel_component_cpu(env, kernel_name, cpu, sources, write)?,
        Kernel::RustCpu(rust) => {
            render_kernel_component_rust(env, kernel_name, &rust.cargo_manifest, write)?
        }
        Kernel::CppCuda(cuda) => {
            render_kernel_component_cuda(env, kernel_name, cuda, sources, write)?
        }
        Kernel::CppRocm(rocm) => {
            render_kernel_component_hip(env, kernel_name, rocm, sources, write)?
        }
        Kernel::CppMetal(metal) => {
            render_kernel_component_metal(env, kernel_name, metal, sources, write)?
        }
        Kernel::CppXpu(xpu) => render_kernel_component_xpu(env, kernel_name, xpu, sources, write)?,
    }

    Ok(())
}

fn render_kernel_component_rust(
    env: &Environment,
    kernel_name: &str,
    cargo_manifest: &str,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/rust-cpu.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                manifest_path => cargo_manifest,
                name => kernel_name,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn render_kernel_component_cpu(
    env: &Environment,
    kernel_name: &str,
    kernel: &CppCpu,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/cpu.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags.as_ref().map(|flags| flags.join(";")),
                includes => kernel.include.as_deref().map(prefix_and_join_includes),
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
    kernel: &CppCuda,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/cuda.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                name => kernel_name,
                cuda_capabilities => kernel.cuda_capabilities.as_deref(),
                cuda_flags => kernel.cuda_flags.as_ref().map(|flags| flags.join(";")),
                cuda_minver => kernel.cuda_minver.as_ref().map(ToString::to_string),
                cxx_flags => kernel.cxx_flags.as_ref().map(|flags| flags.join(";")),
                includes => kernel.include.as_deref().map(prefix_and_join_includes),
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
    kernel: &CppRocm,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/hip.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags.as_ref().map(|flags| flags.join(";")),
                rocm_archs => kernel.rocm_archs.as_deref(),
                hip_flags => kernel.hip_flags.as_ref().map(|flags| flags.join(";")),
                includes => kernel.include.as_deref().map(prefix_and_join_includes),
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
    kernel: &CppMetal,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/metal.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags.as_ref().map(|flags| flags.join(";")),
                includes => kernel.include.as_deref().map(prefix_and_join_includes),
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
    kernel: &CppXpu,
    sources: String,
    write: &mut impl Write,
) -> Result<()> {
    env.get_template("kernel-component/xpu.cmake")
        .wrap_err("Cannot get kernel template")?
        .render_captured_to(
            context! {
                cxx_flags => kernel.cxx_flags.as_ref().map(|flags| flags.join(";")),
                sycl_flags => kernel.sycl_flags.as_ref().map(|flags| flags.join(";")),
                includes => kernel.include.as_deref().map(prefix_and_join_includes),
                kernel_name => kernel_name,
                sources => sources,
            },
            &mut *write,
        )
        .wrap_err("Cannot render kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}
