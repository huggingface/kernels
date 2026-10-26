use super::apply_rewrite;
use crate::python::{self, DottedPath};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::Result;

#[derive(Debug)]
pub struct RemapModule {
    pattern: Pattern,
    from: DottedPath,
    to: DottedPath,
    changes: Option<usize>,
}

impl RemapModule {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            pattern: args.take("in")?.parse()?,
            from: args.take("from")?.parse()?,
            to: args.take("to")?.parse()?,
            changes: args.take_usize_opt("changes")?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "rewrote", |path, src| {
            python::remap_source(path, src, &self.from, &self.to)
        })
    }
}
