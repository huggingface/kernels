use super::glob_comma_list;
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Prune {
    keep: Vec<Pattern>,
}

impl Prune {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let keep = glob_comma_list(&args.take("keep")?)
            .iter()
            .map(|s| s.parse())
            .collect::<Result<Vec<Pattern>>>()?;
        if keep.is_empty() {
            bail!("keep must list at least one glob");
        }
        Ok(Self { keep })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let mut kept = std::collections::BTreeSet::new();
        for glob in &self.keep {
            let matches = ws.glob(glob);
            if matches.is_empty() {
                bail!("keep glob {glob:?} matches nothing");
            }
            kept.extend(matches);
        }
        let doomed: Vec<String> = ws
            .glob_str("**")?
            .into_iter()
            .filter(|p| !kept.contains(p))
            .collect();
        let n = doomed.len();
        for path in doomed {
            ws.delete(&path)?;
        }
        Ok(format!("removed {n} file(s), kept {}", kept.len()))
    }
}
