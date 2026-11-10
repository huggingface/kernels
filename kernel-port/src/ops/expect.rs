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
    // The one matched file has this git blob id: an overlay copy was made
    // from exactly this upstream file.
    Blob(String),
}

/// The id git gives a file with these contents.
pub(crate) fn git_blob_id(bytes: &[u8]) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    crate::hex(&hasher.finalize())
}

impl Expect {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let pattern = args.take("in")?.parse()?;
        if let Some(blob) = args.take_opt("blob") {
            if args.take_opt("find").is_some() || args.take_opt("files").is_some() {
                bail!("blob= cannot be combined with find= or files=");
            }
            if blob.len() != 40 || !blob.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
                bail!("blob must be a full 40-character lowercase git blob id, got {blob:?}");
            }
            return Ok(Self {
                pattern,
                kind: ExpectKind::Blob(blob),
            });
        }
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
            ExpectKind::Blob(expected) => {
                let [path] = files.as_slice() else {
                    bail!(
                        "blob= needs {:?} to match exactly one file, found {}",
                        self.pattern,
                        files.len()
                    );
                };
                let actual = git_blob_id(ws.current_bytes(path).unwrap_or_default());
                if actual != *expected {
                    bail!(
                        "{path} is no longer the upstream file its overlay copy was made from \
                         (blob {expected}, now {actual}) - see what changed with `git diff \
                         {expected} {actual}` in the upstream checkout, carry it into the \
                         overlay copy, and update blob="
                    );
                }
                Ok(format!("{path} is blob {}", &actual[..12]))
            }
        }
    }
}
