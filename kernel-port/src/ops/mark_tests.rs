use super::{apply_rewrite_of, comma_list};
use crate::python::{self, TestSelection};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct MarkTests {
    pattern: Pattern,
    marker: String,
    exclude: BTreeSet<String>,
    only: BTreeSet<String>,
    changes: Option<usize>,
}

impl MarkTests {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let names = |v: Option<String>| -> BTreeSet<String> {
            comma_list(&v.unwrap_or_default()).into_iter().collect()
        };
        let op = Self {
            pattern: args.take("in")?.parse()?,
            marker: args.take("marker")?,
            exclude: names(args.take_opt("exclude")),
            only: names(args.take_opt("only")),
            changes: args.take_usize_opt("changes")?,
        };
        if !op.exclude.is_empty() && !op.only.is_empty() {
            bail!("exclude= and only= are mutually exclusive");
        }
        python::validate_marker(&op.marker)?;
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let (select, key) = if self.only.is_empty() {
            (TestSelection::Exclude(self.exclude.clone()), "exclude")
        } else {
            (TestSelection::Only(self.only.clone()), "only")
        };
        let mut named = BTreeSet::new();
        let summary = apply_rewrite_of(
            ws,
            &self.pattern,
            self.changes,
            "marked",
            "test",
            |path, src| python::mark_tests_source(path, src, &self.marker, &select, &mut named),
        )?;
        // A stale name would otherwise let a renamed upstream test be marked,
        // or silently stop being marked.
        let stale: Vec<_> = select.names().difference(&named).collect();
        if !stale.is_empty() {
            bail!("{key} names no test in {:?}: {stale:?}", self.pattern);
        }
        Ok(summary)
    }
}
