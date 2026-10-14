use super::Facts;
use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
enum From {
    // A file, or a directory and everything under it.
    Path(String),
    // Every file matching a glob, renamed through a template.
    Glob { pattern: Pattern, files: usize },
}

#[derive(Debug)]
pub struct Move {
    from: From,
    to: String,
    refs: Option<usize>,
}

fn is_glob(s: &str) -> bool {
    s.contains(['*', '?', '[', '{'])
}

const PLACEHOLDERS: &[&str] = &["{dir}", "{name}", "{stem}", "{ext}"];

// Fill `{dir}`, `{name}`, `{stem}` and `{ext}` from one matched path.
fn render(template: &str, path: &str) -> String {
    let (dir, name) = path.rsplit_once('/').unwrap_or(("", path));
    let (stem, ext) = name.rsplit_once('.').unwrap_or((name, ""));
    let out = template
        .replace("{dir}", dir)
        .replace("{name}", name)
        .replace("{stem}", stem)
        .replace("{ext}", ext);
    // An empty {dir} must not leave a leading slash.
    out.trim_start_matches('/').to_string()
}

impl Move {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let from = args.take("from")?;
        let to = args.take("to")?;
        let files = args.take_usize_opt("files")?;
        let refs = args.take_usize_opt("refs")?;
        let from = if is_glob(&from) {
            let Some(files) = files else {
                bail!("a glob `from` requires files=N, the number of files it moves");
            };
            if !PLACEHOLDERS.iter().any(|p| to.contains(p)) {
                bail!(
                    "a glob `from` needs a `to` template using {}",
                    PLACEHOLDERS.join(", ")
                );
            }
            From::Glob {
                pattern: from.parse()?,
                files,
            }
        } else {
            if files.is_some() {
                bail!("files= only applies when `from` is a glob");
            }
            From::Path(from)
        };
        Ok(Self { from, to, refs })
    }

    pub(super) fn apply(&self, ws: &mut Workspace, facts: &mut Facts) -> Result<String> {
        let before: BTreeSet<String> = ws.paths().into_iter().collect();
        let pairs = match &self.from {
            From::Path(from) => ws.rename(from, &self.to)?,
            From::Glob { pattern, files } => {
                let paths = ws.glob(pattern);
                if paths.len() != *files {
                    bail!(
                        "expected exactly {files} file(s) for {pattern:?}, found {}",
                        paths.len()
                    );
                }
                let pairs: Vec<(String, String)> = paths
                    .into_iter()
                    .map(|p| {
                        let new = render(&self.to, &p);
                        (p, new)
                    })
                    .collect();
                // Check every destination before renaming anything.
                let mut seen = BTreeSet::new();
                for (old, new) in &pairs {
                    if old == new {
                        bail!("{old:?} would move onto itself");
                    }
                    if !seen.insert(new) || before.contains(new) {
                        bail!("destination {new:?} already exists");
                    }
                }
                for (old, new) in &pairs {
                    ws.rename(old, new)?;
                }
                pairs
            }
        };

        // Keep quoted #includes pointing at the files they named.
        let moves: BTreeMap<String, String> = pairs.iter().cloned().collect();
        let (updates, n_refs) = ws.fix_includes(&before, &moves);
        match self.refs {
            Some(expected) if expected != n_refs => bail!(
                "expected exactly {expected} rewritten #include(s) but made {n_refs} - \
                 review the references and update refs="
            ),
            None if n_refs > 0 => bail!(
                "the move breaks {n_refs} #include(s); they are rewritten to the new \
                 locations, pin that with refs={n_refs}"
            ),
            _ => {}
        }
        for (path, text) in updates {
            ws.set_text(&path, text);
        }

        let n = pairs.len();
        facts.moved.extend(pairs);
        let refs = if n_refs > 0 {
            format!(", rewrote {n_refs} #include(s)")
        } else {
            String::new()
        };
        Ok(match &self.from {
            From::Path(_) => format!("moved {n} file(s) to {:?}{refs}", self.to),
            From::Glob { pattern, .. } => {
                format!(
                    "moved {n} file(s) matching {pattern:?} to {:?}{refs}",
                    self.to
                )
            }
        })
    }
}
