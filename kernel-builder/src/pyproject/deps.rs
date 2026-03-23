use std::collections::HashSet;
use std::io::Write;

use eyre::{Context, Result};
use minijinja::{context, Environment};

use crate::config::{Build, Dependency};

pub fn render_deps(env: &Environment, build: &Build, write: &mut impl Write) -> Result<()> {
    // Collect all dependencies.
    let mut deps = HashSet::new();
    for kernel in build.kernels.values() {
        deps.extend(kernel.depends())
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
            Dependency::SyclTla => {
                env.get_template("xpu/dep-sycl-tla.cmake")?
                    .render_to_write(context! {}, &mut *write)?;
            }
            Dependency::MetalCpp => {
                // TODO: add CMake dependency.
            }
            Dependency::Torch => (),
        }
        write.write_all(b"\n")?;
    }

    Ok(())
}
