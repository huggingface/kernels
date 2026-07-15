use std::{io::Write, path::Path};

use eyre::{Context, Result};
use itertools::Itertools;
use kernels_data::config::{Build, Kernel};
use minijinja::{context, Environment};

use crate::pyproject::common::prefix_and_join_includes;

pub fn render_kernel_components(
    env: &Environment,
    build: &Build,
    write: &mut impl Write,
) -> Result<()> {
    for (kernel_name, kernel) in build.kernels.iter().sorted_by(|(a, _), (b, _)| a.cmp(b)) {
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
        Kernel::RustCpu { .. } | Kernel::RustCuda { .. } => {
            render_kernel_component_rust(env, kernel_name, kernel, write)?
        }
    }

    Ok(())
}

fn render_kernel_component_rust(
    env: &Environment,
    kernel_name: &str,
    kernel: &Kernel,
    write: &mut impl Write,
) -> Result<()> {
    let (template, src, lib_name, features, device_manifest, ptx_dir, cuda_capabilities) =
        match kernel {
            Kernel::RustCpu {
                src,
                lib_name,
                features,
                ..
            } => (
                "kernel-component/rust-cpu.cmake",
                src,
                lib_name,
                features,
                None,
                None,
                None,
            ),
            Kernel::RustCuda {
                src,
                lib_name,
                features,
                device_manifest,
                ptx_dir,
                cuda_capabilities,
                ..
            } => (
                "kernel-component/rust-cuda.cmake",
                src,
                lib_name,
                features,
                device_manifest.as_deref(),
                ptx_dir.as_deref(),
                cuda_capabilities.as_ref(),
            ),
            _ => {
                unreachable!("render_kernel_component_rust only accepts Rust kernels")
            }
        };
    let manifest_path = rust_manifest_src(kernel_name, src)?;

    let lib_name = rust_lib_name(manifest_path, lib_name).ok_or_else(|| {
        eyre::eyre!(
            "Rust kernel `{kernel_name}`: cannot derive `lib-name` from \
             `src = [\"{}\"]`, set it explicitly",
            manifest_path
        )
    })?;

    env.get_template(template)
        .wrap_err("Cannot get Rust kernel template")?
        .render_captured_to(
            context! {
                cuda_capabilities => cuda_capabilities,
                device_manifest => device_manifest,
                features => features,
                lib_name => lib_name,
                manifest_path => manifest_path,
                name => kernel_name,
                ptx_dir => ptx_dir,
            },
            &mut *write,
        )
        .wrap_err("Cannot render Rust kernel template")?;

    write.write_all(b"\n")?;

    Ok(())
}

fn rust_manifest_src<'a>(kernel_name: &str, src: &'a [String]) -> Result<&'a str> {
    src.iter()
        .find(|path| {
            Path::new(path.as_str())
                .file_name()
                .is_some_and(|name| name == "Cargo.toml")
        })
        .map(String::as_str)
        .ok_or_else(|| eyre::eyre!("Rust kernel `{kernel_name}`: `src` must include Cargo.toml"))
}

fn rust_lib_name(manifest_path: &str, lib_name: &Option<String>) -> Option<String> {
    lib_name.clone().or_else(|| {
        Path::new(manifest_path)
            .parent()
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().replace('-', "_"))
    })
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
