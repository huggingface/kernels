use super::{apply_rewrite_of, comma_list};
use crate::python;
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct MarkTests {
    pattern: Pattern,
    marker: String,
    exclude: BTreeSet<String>,
    changes: Option<usize>,
}

impl MarkTests {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            marker: args.take("marker")?,
            exclude: comma_list(&args.take_opt("exclude").unwrap_or_default())
                .into_iter()
                .collect(),
            changes: args.take_usize_opt("changes")?,
        };
        python::validate_marker(&op.marker)?;
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let mut excluded = BTreeSet::new();
        let summary = apply_rewrite_of(
            ws,
            &self.pattern,
            self.changes,
            "marked",
            "test",
            |path, src| {
                python::mark_tests_source(path, src, &self.marker, &self.exclude, &mut excluded)
            },
        )?;
        // A stale name would otherwise let a renamed upstream test be marked.
        let stale: Vec<_> = self.exclude.difference(&excluded).collect();
        if !stale.is_empty() {
            bail!("exclude names no test in {:?}: {stale:?}", self.pattern);
        }
        Ok(summary)
    }
}
