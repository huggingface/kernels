use super::apply_rewrite;
use crate::python::{self, DottedPath};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct KernelizeImports {
    pattern: Pattern,
    package: String,
    binding: String,
    kernel: String,
    version: usize,
    check_arch: bool,
    changes: Option<usize>,
}

impl KernelizeImports {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let package = args.take("package")?;
        let parsed: DottedPath = package.parse()?;
        if parsed.parts().len() != 1 {
            bail!("package must be one top-level Python name, got {package:?}");
        }
        let kernel = args.take("kernel")?;
        if kernel.is_empty() {
            bail!("kernel must not be empty");
        }
        let binding = args.take_opt("binding").unwrap_or_else(|| package.clone());
        // `get_kernel` checks the device architecture by default; kernels
        // built only for newer devices skip it so their tests can be
        // collected, and gated, everywhere.
        let check_arch = match args.take_opt("check_arch").as_deref() {
            None | Some("true") => true,
            Some("false") => false,
            Some(other) => bail!("check_arch must be #true or #false, got {other:?}"),
        };
        Ok(Self {
            pattern: args.take("in")?.parse()?,
            package,
            binding,
            kernel,
            version: args.take_usize("version")?,
            check_arch,
            changes: args.take_usize_opt("changes")?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(
            ws,
            &self.pattern,
            self.changes,
            "kernelized",
            |path, src| {
                python::kernelize_imports_source(
                    path,
                    src,
                    &self.package,
                    &self.binding,
                    &self.kernel,
                    self.version,
                    self.check_arch,
                )
            },
        )
    }
}
