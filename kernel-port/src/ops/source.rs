use super::git::{CommitSha, check_clean_checkout, git, normalize_git_url};
use super::{Facts, Inputs};
use crate::recipe::Args;
use anyhow::{Context, Result, bail};

#[derive(Debug)]
pub struct Source {
    repo: String,
    commit: CommitSha,
}

impl Source {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            repo: args.take("repo")?,
            commit: args.take("commit")?.parse()?,
        })
    }

    pub(super) fn apply(&self, inputs: &Inputs, facts: &mut Facts) -> Result<String> {
        let head = git(&inputs.root, &["rev-parse", "HEAD"]).with_context(|| {
            format!(
                "source pinning requires --dir to be a git checkout of {}",
                self.repo
            )
        })?;
        if head != self.commit.as_str() {
            bail!(
                "checkout is at {head} but this port is written against {} - \
                 re-clone at the pinned commit, or re-verify the port and update the pin",
                self.commit
            );
        }
        // A checkout with no origin still passes; only a mismatch fails.
        if let Ok(origin) = git(&inputs.root, &["remote", "get-url", "origin"])
            && normalize_git_url(&origin) != normalize_git_url(&self.repo)
        {
            bail!(
                "checkout origin is {origin} but this port is for {} - wrong repository",
                self.repo
            );
        }
        check_clean_checkout(&inputs.root, "source")?;
        facts.record_source("upstream", &self.repo, self.commit.as_str());
        Ok(format!("verified {} @ {}", self.repo, &head[..12]))
    }
}
