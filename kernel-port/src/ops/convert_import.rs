use super::apply_rewrite;
use crate::python::{self, DottedPath};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::Result;

#[derive(Debug)]
pub struct ConvertImport {
    pattern: Pattern,
    prefix: DottedPath,
    changes: Option<usize>,
}

impl ConvertImport {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            pattern: args.take("in")?.parse()?,
            prefix: args.take("prefix")?.parse()?,
            changes: args.take_usize_opt("changes")?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "converted", |path, src| {
            python::convert_imports_source(path, src, &self.prefix)
        })
    }
}
