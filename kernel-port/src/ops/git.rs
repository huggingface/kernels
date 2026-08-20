use anyhow::{Context, Result, bail};
use std::path::Path;

#[derive(Debug)]
pub struct CommitSha(String);

impl CommitSha {
    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for CommitSha {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() != 40 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("commit must be a full 40-character hex SHA, got {s:?}");
        }
        Ok(Self(s.to_string()))
    }
}

impl std::fmt::Display for CommitSha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub(super) fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .context("running git")?;
    if !out.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            root.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// So a pin written as .../repo.git matches a checkout cloned from .../repo.
pub(super) fn normalize_git_url(url: &str) -> String {
    url.trim_end_matches('/')
        .trim_end_matches(".git")
        .to_string()
}

pub(super) fn check_clean_checkout(root: &Path, what: &str) -> Result<()> {
    let status = git(root, &["status", "--porcelain"])?;
    if !status.is_empty() {
        bail!(
            "{what} checkout {} is not clean (modified or untracked files present); \
             the output would not be reproducible from the pin:\n{status}",
            root.display()
        );
    }
    Ok(())
}
