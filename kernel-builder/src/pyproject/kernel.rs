use std::io::Write;

use eyre::{Context, Result};
use itertools::Itertools;
use kernels_common::config::{Build, Kernel};
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
        Kernel::CppCpu { .. } => {
            render_kernel_component_cpu(env, kernel_name, kernel, sources, write)?
        }
        Kernel::RustCpu { cargo_manifest, .. } => {
            render_kernel_component_rust(env, kernel_name, cargo_manifest, write)?
        }
        Kernel::RustCuda {
            cargo_manifest,
            cuda_capabilities,
            ..
        } => {
            env.get_template("kernel-component/rust-cuda.cmake")?
                .render_captured_to(
                    context! {
                        name => kernel_name,
                        manifest_path => cargo_manifest,
                        cuda_capabilities => cuda_capabilities,
                    },
                    &mut *write,
                )?;
            write.write_all(b"\n")?;
        }
        Kernel::CppCuda { .. } => {
            render_kernel_component_cuda(env, kernel_name, kernel, sources, write)?
        }
        Kernel::CppRocm { .. } => {
            render_kernel_component_hip(env, kernel_name, kernel, sources, write)?
        }
        Kernel::CppMetal { .. } => {
            render_kernel_component_metal(env, kernel_name, kernel, sources, write)?
        }
        Kernel::CppXpu { .. } => {
            render_kernel_component_xpu(env, kernel_name, kernel, sources, write)?
        }
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
        Kernel::CppCuda {
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
        Kernel::CppRocm {
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
        Kernel::CppXpu { sycl_flags, .. } => sycl_flags.as_deref(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_components_render_for_their_backend() {
        let mut env = Environment::new();
        minijinja_embed::load_templates!(&mut env);
        for (kernel, backend, capabilities) in [
            (
                Kernel::RustCpu {
                    cargo_manifest: "host/Cargo.toml".into(),
                    depends: vec![],
                    src: vec![],
                },
                "CPU",
                None,
            ),
            (
                Kernel::RustCuda {
                    cargo_manifest: "host/Cargo.toml".into(),
                    cuda_capabilities: Some(vec!["8.0".into(), "9.0".into()]),
                    depends: vec![],
                    src: vec![],
                },
                "CUDA",
                Some("CUDA_CAPABILITIES 8.0 9.0"),
            ),
        ] {
            let mut rendered = vec![];
            render_kernel_component(&env, "example", &kernel, &mut rendered).unwrap();
            let rendered = String::from_utf8(rendered).unwrap();
            assert!(rendered.contains(&format!("if(GPU_LANG STREQUAL \"{backend}\")")));
            assert!(rendered.contains("MANIFEST_PATH \"host/Cargo.toml\""));
            assert_eq!(
                rendered.contains("CUDA_CAPABILITIES"),
                capabilities.is_some()
            );
            if let Some(capabilities) = capabilities {
                assert!(rendered.contains(capabilities));
            }
        }
    }
}
