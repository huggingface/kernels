use super::apply_rewrite;
use crate::python;
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::Result;

#[derive(Debug)]
pub struct EnsureImport {
    pattern: Pattern,
    from: String,
    name: String,
    changes: Option<usize>,
}

impl EnsureImport {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            from: args.take("from")?,
            name: args.take("name")?,
            changes: args.take_usize_opt("changes")?,
        };
        python::validate_ensure_import(&op.from, &op.name)?;
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "added", |path, src| {
            python::ensure_import_source(path, src, &self.from, &self.name)
        })
    }
}
