use super::check_changes_pin;
use crate::recipe::Args;
use crate::workspace::Workspace;
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct EnsureInit {
    under: String,
    changes: Option<usize>,
}

impl EnsureInit {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            under: args.take("under")?.trim_matches('/').to_string(),
            changes: args.take_usize_opt("changes")?,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let files = ws.glob_str(&format!("{}/**", self.under))?;
        if files.is_empty() {
            bail!("no files under {:?}", self.under);
        }
        let mut dirs = std::collections::BTreeSet::new();
        dirs.insert(self.under.clone());
        for path in &files {
            if !crate::is_python(path) {
                continue;
            }
            let mut dir = path.rsplit_once('/').map(|(d, _)| d.to_string());
            while let Some(d) = dir {
                if d.len() <= self.under.len() {
                    break;
                }
                dirs.insert(d.clone());
                dir = d.rsplit_once('/').map(|(p, _)| p.to_string());
            }
        }
        let mut added = 0;
        for dir in dirs {
            let init = format!("{dir}/__init__.py");
            if ws.current_bytes(&init).is_none() {
                ws.insert(&init, Vec::new());
                added += 1;
            }
        }
        check_changes_pin(self.changes, added)?;
        Ok(format!("added {added} missing __init__.py file(s)"))
    }
}
