use crate::recipe::Args;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct Expect {
    pattern: Pattern,
    kind: ExpectKind,
}

#[derive(Debug)]
enum ExpectKind {
    Text { find: String, count: usize },
    FileCount(usize),
}

impl Expect {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let pattern = args.take("in")?.parse()?;
        let kind = match (args.take_opt("find"), args.take_usize_opt("files")?) {
            (Some(find), None) => {
                if find.is_empty() {
                    bail!("find must not be empty");
                }
                ExpectKind::Text {
                    find,
                    count: args.take_usize("count")?,
                }
            }
            (None, Some(n)) => ExpectKind::FileCount(n),
            (Some(_), Some(_)) => bail!("find and files are mutually exclusive"),
            (None, None) => bail!("expect requires either find=.../count=N or files=N"),
        };
        Ok(Self { pattern, kind })
    }

    pub(super) fn apply(&self, ws: &Workspace) -> Result<String> {
        let files = ws.glob(&self.pattern);
        match &self.kind {
            ExpectKind::FileCount(expected) => {
                if files.len() != *expected {
                    bail!(
                        "expected {:?} to match exactly {expected} file(s), found {} - \
                         the upstream file set drifted; update the port definition ({})",
                        self.pattern,
                        files.len(),
                        files.join(", ")
                    );
                }
                Ok(format!("{expected} file(s), as expected"))
            }
            ExpectKind::Text { find, count } => {
                if files.is_empty() {
                    bail!("{:?} matches no files", self.pattern);
                }
                let mut found = 0;
                for path in &files {
                    found += ws.get_text(path)?.matches(find).count();
                }
                if found != *count {
                    bail!(
                        "expected exactly {count} occurrence(s) of {find:?} in {:?}, found \
                         {found} - upstream changed text this port depends on; update the \
                         port definition",
                        self.pattern
                    );
                }
                Ok(format!("{found} occurrence(s), as expected"))
            }
        }
    }
}
