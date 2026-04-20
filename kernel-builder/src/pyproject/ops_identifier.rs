use std::path::Path;

use eyre::{Result, WrapErr};
use git2::Repository;
use kernels_data::config::Backend;
use rand::Rng;

pub fn random_identifier() -> String {
    // Generate a random string when no ops_id is provided
    let mut rng = rand::thread_rng();
    let build_id: u64 = rng.gen();
    base32::encode(
        base32::Alphabet::Rfc4648Lower { padding: false },
        &build_id.to_le_bytes(),
    )
}

pub fn git_identifier(target_dir: impl AsRef<Path>) -> Result<String> {
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

pub struct KernelIdentifier {
    name: String,
    unique_id: String,
}

impl KernelIdentifier {
    /// Create a kernel identifier.
    ///
    /// Create an identifier for the kernel in `target_dir` with the given
    /// `name`. `unique_id` must be a unique identifier for the kernel
    /// source revision. If this identefier is not provided, a Git short
    /// hash is extracted from the repository of `target_dir`. If that
    /// fails, a random identifier is generated.
    pub fn new(kernel_dir: impl AsRef<Path>, name: String, unique_id: Option<String>) -> Self {
        let unique_id = unique_id.unwrap_or_else(|| match git_identifier(kernel_dir.as_ref()) {
            Ok(rev) => rev,
            Err(_) => random_identifier(),
        });

        Self { name, unique_id }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create the kernel identifier string for a given backend.
    pub fn to_string_for_backend(&self, backend: Backend) -> String {
        format!("_{}_{}_{}", self.name, backend, self.unique_id)
    }

    pub fn unique_id(&self) -> &str {
        &self.unique_id
    }
}
