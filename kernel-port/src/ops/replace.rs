use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Replace {
    pattern: Pattern,
    find: String,
    with: String,
    count: Option<usize>,
}

impl Replace {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            find: args.take("find")?,
            with: args.take("with")?,
            count: args.take_usize_opt("count")?,
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
            per_file.push((path.clone(), n));
            found += n;
        }
        if let Some(count) = self.count
            && found != count
        {
            bail!(
                "expected exactly {} match(es) of {:?} in {:?}, found {} ({})",
                count,
                self.find,
                self.pattern,
                found,
                if found == 0 {
                    "no files matched the text".to_string()
                } else {
                    per_file
                        .iter()
                        .filter(|(_, n)| *n > 0)
                        .map(|(p, n)| format!("{p}: {n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
        }
        if self.count.is_none() {
            let mismatches = per_file
                .iter()
                .filter(|(_, n)| *n != 1)
                .map(|(path, n)| format!("{path}: {n}"))
                .collect::<Vec<_>>();
            if !mismatches.is_empty() {
                bail!(
                    "expected exactly one match of {:?} in every file matched by {:?} ({})",
                    self.find,
                    self.pattern,
                    mismatches.join(", ")
                );
            }
        }
        let n_files = per_file.iter().filter(|(_, n)| *n > 0).count();
        for (path, n) in per_file {
            if n == 0 {
                continue;
            }
            let updated = ws.get_text(&path)?.replace(&self.find, &self.with);
            ws.set_text(&path, updated);
        }
        Ok(format!("{found} replacement(s) in {n_files} file(s)"))
    }
}
