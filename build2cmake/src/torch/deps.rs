use std::collections::HashSet;
use std::io::Write;

use eyre::{Context, Result};
use minijinja::{context, Environment};

use crate::config::{Backend, Build, Dependency};

pub fn render_deps(
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
            Dependency::CutlassSycl => {
                env.get_template("xpu/dep-cutlass-sycl.cmake")?
                    .render_to_write(context! {}, &mut *write)?;
            }
            Dependency::Torch => (),
            _ => {
                eprintln!("Warning: {backend:?} backend doesn't need/support dependency: {dep:?}");
            }
        }
        write.write_all(b"\n")?;
    }

    Ok(())
}
