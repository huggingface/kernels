use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Delete {
    pattern: Pattern,
}

impl Delete {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            pattern: args.take("in")?.parse()?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let matches = ws.glob(&self.pattern);
        if matches.is_empty() {
            bail!("{:?} matches nothing", self.pattern);
        }
        let n = matches.len();
        for path in matches {
            ws.delete(&path)?;
        }
        Ok(format!("removed {n} file(s)"))
    }
}
