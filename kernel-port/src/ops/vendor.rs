use super::git::{CommitSha, check_clean_checkout, git, normalize_git_url};
use super::{Facts, Inputs, copy_tree};
use crate::recipe::Args;
use crate::workspace::Workspace;
use anyhow::{Context, Result, bail};

#[derive(Debug)]
pub struct Vendor {
    name: String,
    repo: String,
    commit: CommitSha,
    path: String,
    to: String,
}

impl Vendor {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            name: args.take("name")?,
            repo: args.take("repo")?,
            commit: args.take("commit")?.parse()?,
            path: args.take("path")?.trim_matches('/').to_string(),
            to: args.take("to")?.trim_matches('/').to_string(),
        })
    }

    pub(super) fn apply(
        &self,
        ws: &mut Workspace,
        inputs: &Inputs,
        facts: &mut Facts,
    ) -> Result<String> {
        let Some(checkout) = inputs.vendors.get(&self.name) else {
            bail!(
                "no checkout supplied for vendor {:?}; pass --vendor {}=<dir> on the CLI",
                self.name,
                self.name
            );
        };
        let head = git(checkout, &["rev-parse", "HEAD"]).with_context(|| {
            format!(
                "vendor {:?} must be a git checkout of {}",
                self.name, self.repo
            )
        })?;
        if head != self.commit.as_str() {
            bail!(
                "vendor {:?} checkout is at {head} but this port is written against {}",
                self.name,
                self.commit
            );
        }
        if let Ok(origin) = git(checkout, &["remote", "get-url", "origin"])
            && normalize_git_url(&origin) != normalize_git_url(&self.repo)
        {
            bail!(
                "vendor {:?} checkout origin is {origin} but this port expects {}",
                self.name,
                self.repo
            );
        }
        check_clean_checkout(checkout, "vendor")?;
        facts.record_source(&self.name, &self.repo, self.commit.as_str());

        let src_root = checkout.join(&self.path);
        if !src_root.is_dir() {
            bail!(
                "{} is not a directory in the vendor checkout",
                src_root.display()
            );
        }
        let copied = copy_tree(ws, &src_root, |rel| format!("{}/{rel}", self.to), true)?;
        if copied == 0 {
            bail!("{} contains no files", src_root.display());
        }
        Ok(format!(
            "vendored {copied} file(s) from {}:{} @ {}",
            self.name,
            self.path,
            &head[..12]
        ))
    }
}
