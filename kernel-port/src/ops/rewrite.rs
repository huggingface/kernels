use crate::recipe::Args;
use crate::rewrite::{Lang, Rule};
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

pub struct Rewrite {
    pattern: Pattern,
    rule: Rule,
    changes: usize,
}

impl std::fmt::Debug for Rewrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rewrite")
            .field("in", &self.pattern)
            .finish_non_exhaustive()
    }
}

impl Rewrite {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let pattern = args.take("in")?.parse()?;
        let rule = Rule::new(&args.take("pattern")?, &args.take("with")?)?;
        // Unlike replace, the pin is mandatory: a structural pattern is still
        // a claim about upstream text, and the count is what makes it checked.
        let changes = args.take_usize("changes")?;
        Ok(Self {
            pattern,
            rule,
            changes,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let paths = ws.glob(&self.pattern);
        if paths.is_empty() {
            bail!("{:?} matches no files", self.pattern);
        }
        let mut updates = Vec::new();
        let mut found_in = Vec::new();
        let mut total = 0;
        for path in paths {
            let text = ws.get_text(&path)?;
            let (new_text, n) = self.rule.apply(text, Lang::of(&path));
            if n > 0 {
                found_in.push(format!("{path}: {n}"));
                updates.push((path, new_text));
                total += n;
            }
        }
        if total != self.changes {
            bail!(
                "expected exactly {} match(es) but found {total} ({}) - upstream drifted; \
                 review the matches and update changes=",
                self.changes,
                if found_in.is_empty() {
                    "none".to_string()
                } else {
                    found_in.join(", ")
                }
            );
        }
        let n_files = updates.len();
        for (path, text) in updates {
            ws.set_text(&path, text);
        }
        Ok(format!("rewrote {total} match(es) in {n_files} file(s)"))
    }
}
