use super::Facts;
use crate::recipe::Args;
use crate::workspace::Workspace;
use anyhow::Result;

#[derive(Debug)]
pub struct Move {
    from: String,
    to: String,
}

impl Move {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            from: args.take("from")?,
            to: args.take("to")?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace, facts: &mut Facts) -> Result<String> {
        let pairs = ws.rename(&self.from, &self.to)?;
        let n = pairs.len();
        facts.moved.extend(pairs);
        Ok(format!("moved {n} file(s) to {:?}", self.to))
    }
}
