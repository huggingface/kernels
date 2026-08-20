use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Replace {
    pattern: Pattern,
    find: String,
    with: String,
    count: usize,
}

impl Replace {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            find: args.take("find")?,
            with: args.take("with")?,
            count: args.take_usize("count")?,
        };
        if op.find.is_empty() {
            bail!("find must not be empty");
        }
        Ok(op)
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let files = ws.glob(&self.pattern);
        if files.is_empty() {
            bail!("{:?} matches no files", self.pattern);
        }
        let mut found = 0;
        let mut per_file = Vec::new();
        for path in &files {
            let n = ws.get_text(path)?.matches(&self.find).count();
            if n > 0 {
                per_file.push((path.clone(), n));
            }
            found += n;
        }
        if found != self.count {
            bail!(
                "expected exactly {} match(es) of {:?} in {:?}, found {} ({})",
                self.count,
                self.find,
                self.pattern,
                found,
                if per_file.is_empty() {
                    "no files matched the text".to_string()
                } else {
                    per_file
                        .iter()
                        .map(|(p, n)| format!("{p}: {n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
        }
        let n_files = per_file.len();
        for (path, _) in per_file {
            let updated = ws.get_text(&path)?.replace(&self.find, &self.with);
            ws.set_text(&path, updated);
        }
        Ok(format!("{found} replacement(s) in {n_files} file(s)"))
    }
}
