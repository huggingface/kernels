use std::path::Path;

use eyre::{Result, WrapErr};
use git2::Repository;
use rand::Rng;

fn random_identifier() -> String {
    // Generate a random string when no ops_id is provided
    let mut rng = rand::thread_rng();
    let build_id: u64 = rng.gen();
    base32::encode(
        base32::Alphabet::Rfc4648Lower { padding: false },
        &build_id.to_le_bytes(),
    )
}

fn git_identifier(target_dir: impl AsRef<Path>) -> Result<String> {
    let repo = Repository::discover(target_dir.as_ref()).context("Cannot open git repository")?;
    let head = repo.head()?;
    let commit = head.peel_to_commit()?;
    let rev = commit.tree_id().to_string().chars().take(7).collect();

    let mut status_options = git2::StatusOptions::new();
    status_options.include_untracked(false); // Ignore untracked files (like generated CMake files)
    status_options.exclude_submodules(true);
    let dirty = !repo.statuses(Some(&mut status_options))?.is_empty();
    Ok(if dirty { format!("{rev}_dirty") } else { rev })
}

pub fn kernel_ops_identifier(
    target_dir: impl AsRef<Path>,
    name: &str,
    ops_id: Option<String>,
) -> String {
    let identifier = ops_id.unwrap_or_else(|| match git_identifier(target_dir.as_ref()) {
        Ok(rev) => rev,
        Err(_) => random_identifier(),
    });

    format!("_{name}_{identifier}")
}
