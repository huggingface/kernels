use std::collections::HashSet;
use std::io::Write;

use eyre::{bail, Context, Result};
use minijinja::{context, Environment};

use kernels_common::config::{Build, Dependency};

pub fn render_deps(env: &Environment, build: &Build, write: &mut impl Write) -> Result<()> {
    // Collect all dependencies.
    let mut deps = HashSet::new();
    for kernel in build.kernels.values() {
        deps.extend(kernel.depends())
    }

    // Only the sycl-tla setup composes the AOT options, do not drop the
    // GRF size silently.
    let cri_grf_size = build.general.xpu.as_ref().and_then(|xpu| xpu.cri_grf_size);
    if cri_grf_size.is_some() && !deps.contains(&Dependency::SyclTla) {
        bail!("`general.xpu.cri-grf-size` needs a kernel that depends on `sycl-tla`");
    }

    for dep in deps {
        match dep {
            Dependency::Cutlass2_10 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_captured_to(
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
                    .render_captured_to(
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
                    .render_captured_to(
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
                    .render_captured_to(
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
                    .render_captured_to(
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
                    .render_captured_to(
                        context! {
                            version => "4.0.0",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::Cutlass4_5 => {
                env.get_template("cuda/dep-cutlass.cmake")
                    .wrap_err("Cannot get CUTLASS dependency template")?
                    .render_captured_to(
                        context! {
                            version => "4.5.2",
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render CUTLASS dependency template")?;
            }
            Dependency::SyclTla => {
                // An Xe35-only TU puts MXFP code that ocloc rejects elsewhere into the
                // generic image, so that image can only lower to cri and pvc/bmg need
                // alias targets. Otherwise one image serves all three, halving the
                // device compilations per TU.
                //
                // The SYCL runtime takes the last AOT image that matches a
                // device, and a spir64_gen image matches every Intel GPU, so
                // spir64_gen must come before the aliases. spir64 is the JIT
                // image for other GPUs.
                let cri_isolated_image = build.kernels.values().any(|kernel| {
                    kernel.sycl_flags().is_some_and(|flags| {
                        flags
                            .iter()
                            .any(|flag| flag.contains("__SYCL_TARGET_INTEL_GPU_CRI__"))
                    })
                });
                let (sycl_aot_devices, sycl_offload_targets) = if cri_isolated_image {
                    ("cri", "spir64_gen,intel_gpu_pvc,intel_gpu_bmg_g21,spir64")
                } else {
                    ("pvc,bmg-g21,cri", "spir64_gen,spir64")
                };
                // The GRF size is an ocloc option for the whole image, which
                // crashes ocloc for pvc and bmg-g21, so it needs the cri-only
                // image.
                let sycl_aot_internal_options = match cri_grf_size {
                    None => None,
                    Some(size) if !matches!(size, 128 | 256 | 512) => {
                        bail!("`general.xpu.cri-grf-size` must be 128, 256, or 512, got {size}")
                    }
                    Some(_) if !cri_isolated_image => bail!(
                        "`general.xpu.cri-grf-size` needs a kernel built for CRI only \
                         (`-D__SYCL_TARGET_INTEL_GPU_CRI__=1` in its `sycl-flags`)"
                    ),
                    Some(size) => Some(format!("-cl-intel-{size}-GRF-per-thread")),
                };
                env.get_template("xpu/dep-sycl-tla.cmake")
                    .wrap_err("Cannot get SYCL TLA dependency template")?
                    .render_captured_to(
                        context! {
                            sycl_aot_devices,
                            sycl_offload_targets,
                            sycl_aot_internal_options,
                        },
                        &mut *write,
                    )
                    .wrap_err("Cannot render SYCL TLA dependency template")?;
            }
            Dependency::MetalCpp => {
                // TODO: add CMake dependency.
            }
            Dependency::Torch => (),
            _ => (),
        }
        write.write_all(b"\n")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(general_xpu: &str, cri_kernel: bool) -> Result<String> {
        let cri_flags = if cri_kernel {
            r#"sycl-flags = ["-fsycl-targets=spir64_gen", "-D__SYCL_TARGET_INTEL_GPU_CRI__=1"]"#
        } else {
            ""
        };
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("build.toml"),
            format!(
                r#"
                [general]
                name = "xpu-demo"
                version = 1
                edition = 6
                license = "Apache-2.0"
                backends = ["xpu"]

                {general_xpu}

                [torch]
                src = ["torch-ext/torch_binding.cpp"]

                [kernel.demo_xpu]
                language = "cpp-xpu"
                depends = ["torch", "sycl_tla"]
                src = ["demo/demo.cpp"]
                {cri_flags}
                "#
            ),
        )
        .unwrap();
        let build = Build::open(tmp.path()).unwrap();

        let mut env = Environment::new();
        minijinja_embed::load_templates!(&mut env);
        let mut rendered = vec![];
        render_deps(&env, &build, &mut rendered)?;
        Ok(String::from_utf8(rendered).unwrap())
    }

    #[test]
    fn cri_grf_size_sets_ocloc_internal_options() {
        let rendered = render("[general.xpu]\ncri-grf-size = 512", true).unwrap();
        assert!(rendered.contains(r#"set(SYCL_AOT_DEVICES "cri")"#));
        assert!(rendered
            .contains(r#"set(SYCL_OFFLOAD_TARGETS "spir64_gen,intel_gpu_pvc,intel_gpu_bmg_g21,spir64")"#));
        assert!(
            rendered.contains(r#"set(SYCL_AOT_INTERNAL_OPTIONS "-cl-intel-512-GRF-per-thread")"#)
        );
    }

    #[test]
    fn no_cri_grf_size_keeps_ocloc_defaults() {
        let rendered = render("", true).unwrap();
        assert!(!rendered.contains("SYCL_AOT_INTERNAL_OPTIONS"));
    }

    #[test]
    fn cri_grf_size_needs_cri_only_image() {
        let err = render("[general.xpu]\ncri-grf-size = 512", false).unwrap_err();
        assert!(err
            .to_string()
            .contains("needs a kernel built for CRI only"));
    }

    #[test]
    fn cri_grf_size_rejects_unknown_size() {
        let err = render("[general.xpu]\ncri-grf-size = 384", true).unwrap_err();
        assert!(err.to_string().contains("must be 128, 256, or 512"));
    }
}
