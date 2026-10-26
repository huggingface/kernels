use super::copy_tree;
use crate::recipe::Args;
use crate::workspace::Workspace;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Overlay {
    source: PathBuf,
}

impl Overlay {
    pub(super) fn build(args: &mut Args, recipe_dir: &Path) -> Result<Self> {
        Ok(Self {
            source: recipe_dir.join(args.take("from")?),
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        if !self.source.is_dir() {
            bail!("{} is not a directory", self.source.display());
        }
        let copied = copy_tree(ws, &self.source, std::string::ToString::to_string, false)?;
        if copied == 0 {
            bail!("{} contains no files", self.source.display());
        }
        Ok(format!("copied {copied} file(s)"))
    }
}
