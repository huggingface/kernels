use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct StripSuffix {
    pattern: Pattern,
    suffix: String,
    files: usize,
}

impl StripSuffix {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            suffix: args.take("suffix")?,
            files: args.take_usize("files")?,
        };
        if op.suffix.is_empty() {
            bail!("suffix must not be empty");
        }
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let paths = ws.glob(&self.pattern);
        if paths.is_empty() {
            bail!("{:?} matches no files", self.pattern);
        }
        if paths.len() != self.files {
            bail!(
                "expected exactly {} file(s) for {:?}, found {}",
                self.files,
                self.pattern,
                paths.len()
            );
        }
        // Every file is checked before any is written, so a partial strip
        // cannot land.
        let mut updates = Vec::with_capacity(paths.len());
        for path in paths {
            let text = ws.get_text(&path)?;
            let Some(stripped) = text.strip_suffix(&self.suffix) else {
                bail!("{path:?} does not end with pinned suffix {:?}", self.suffix);
            };
            updates.push((path, stripped.to_string()));
        }
        for (path, updated) in updates {
            ws.set_text(&path, updated);
        }
        Ok(format!("stripped suffix from {} file(s)", self.files))
    }
}
