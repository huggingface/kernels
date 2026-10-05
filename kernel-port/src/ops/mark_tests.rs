use super::apply_rewrite_of;
use crate::python;
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::Result;

#[derive(Debug)]
pub struct MarkTests {
    pattern: Pattern,
    marker: String,
    changes: Option<usize>,
}

impl MarkTests {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            marker: args.take("marker")?,
            changes: args.take_usize_opt("changes")?,
        };
        python::validate_marker(&op.marker)?;
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite_of(
            ws,
            &self.pattern,
            self.changes,
            "marked",
            "test",
            |path, src| python::mark_tests_source(path, src, &self.marker),
        )
    }
}
