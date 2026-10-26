use super::apply_rewrite;
use crate::python::{self};
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct RelativizeImports {
    pattern: Pattern,
    package_root: String,
    changes: Option<usize>,
    root_relative: bool,
}

impl RelativizeImports {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            pattern: args.take("in")?.parse()?,
            package_root: args.take("package_root")?.trim_end_matches('/').to_string(),
            changes: args.take_usize_opt("changes")?,
            root_relative: match args.take_opt("root_relative").as_deref() {
                None | Some("false") => false,
                Some("true") => true,
                Some(other) => bail!("root_relative must be true or false, got {other:?}"),
            },
        })
    }

    fn package_of(&self, path: &str) -> Result<Vec<String>> {
        let parent = self
            .package_root
            .rfind('/')
            .map_or("", |i| &self.package_root[..=i]);
        let Some(rel) = path.strip_prefix(parent) else {
            bail!("{path:?} is not under package_root {:?}", self.package_root);
        };
        let mut components: Vec<String> = rel.split('/').map(str::to_string).collect();
        components.pop();
        if components.is_empty() {
            bail!("{path:?} sits at the package root's parent; it has no package");
        }
        Ok(components)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "rewrote", |path, src| {
            let package = self.package_of(path)?;
            if self.root_relative {
                python::relativize_source_from_root(path, src, &package)
            } else {
                python::relativize_source(path, src, &package)
            }
        })
    }
}
