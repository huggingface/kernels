use super::apply_rewrite;
use crate::python::{self, DottedPath};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct KernelizeImports {
    pattern: Pattern,
    package: String,
    kernel: String,
    version: usize,
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
        Ok(Self {
            pattern: args.take("in")?.parse()?,
            package,
            kernel,
            version: args.take_usize("version")?,
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
                    &self.kernel,
                    self.version,
                )
            },
        )
    }
}
