use std::path::Path;

use eyre::{Result, WrapErr};
use git2::Repository;
use kernels_data::config::Backend;
use kernels_data::metadata::{GitInfo, KernelBuilderInfo};
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

pub fn git_info(target_dir: impl AsRef<Path>) -> Option<GitInfo> {
    let repo = Repository::discover(target_dir.as_ref()).ok()?;
    let head = repo.head().ok()?;
    let commit = head.peel_to_commit().ok()?;
    let sha = commit.id().to_string();

    let mut status_options = git2::StatusOptions::new();
    status_options.include_untracked(false); // Ignore untracked files (like generated CMake files)
    status_options.exclude_submodules(true);
    let dirty = !repo.statuses(Some(&mut status_options)).ok()?.is_empty();

    Some(GitInfo { sha, dirty })
}

pub struct KernelIdentifier {
    name: String,
    unique_id: String,
    git_info: Option<GitInfo>,
    kernel_builder: Option<KernelBuilderInfo>,
}

impl KernelIdentifier {
    /// Create a kernel identifier.
    ///
    /// Create an identifier for the kernel in `target_dir` with the given
    /// `name`. `unique_id` must be a unique identifier for the kernel
    /// source revision. If this identefier is not provided, a Git short
    /// hash is extracted from the repository of `target_dir`. If that
    /// fails, a random identifier is generated.
    pub fn new(
        kernel_dir: impl AsRef<Path>,
        name: String,
        unique_id: Option<String>,
        git_override: Option<GitInfo>,
        kernel_builder: Option<KernelBuilderInfo>,
    ) -> Self {
        // Prefer an explicitly provided git provenance (e.g. passed by Nix
        // builds, where the source tree has no `.git`); fall back to detecting
        // it from the kernel's git repository.
        let git_info = git_override.or_else(|| git_info(kernel_dir.as_ref()));
        let unique_id = unique_id.unwrap_or_else(|| match git_identifier(kernel_dir.as_ref()) {
            Ok(rev) => rev,
            Err(_) => random_identifier(),
        });

        Self {
            name,
            unique_id,
            git_info,
            kernel_builder,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn git_info(&self) -> Option<&GitInfo> {
        self.git_info.as_ref()
    }

    /// Externally-provided `kernel-builder` provenance, when available. When
    /// `None`, the compile-time baked provenance is used instead.
    pub fn kernel_builder(&self) -> Option<&KernelBuilderInfo> {
        self.kernel_builder.as_ref()
    }

    /// Create the kernel identifier string for a given backend.
    pub fn to_string_for_backend(&self, backend: Backend) -> String {
        format!("_{}_{}_{}", self.name, backend, self.unique_id)
    }

    pub fn unique_id(&self) -> &str {
        &self.unique_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_provenance_overrides_are_stored() {
        let tmp = tempfile::tempdir().unwrap();
        let git = GitInfo {
            sha: "a".repeat(40),
            dirty: true,
        };
        let kernel_builder = KernelBuilderInfo {
            version: "0.16.0-dev0".to_owned(),
            sha: Some("b".repeat(40)),
            dirty: false,
        };

        let id = KernelIdentifier::new(
            tmp.path(),
            "relu".to_owned(),
            Some("rev123".to_owned()),
            Some(git),
            Some(kernel_builder),
        );

        assert_eq!(id.unique_id(), "rev123");
        assert_eq!(id.to_string_for_backend(Backend::Cuda), "_relu_cuda_rev123");

        let git = id.git_info().expect("kernel git info");
        assert_eq!(git.sha, "a".repeat(40));
        assert!(git.dirty);

        let kb = id.kernel_builder().expect("kernel-builder info");
        assert_eq!(kb.sha.as_deref(), Some(&"b".repeat(40)[..]));
        assert!(!kb.dirty);
    }

    #[test]
    fn no_provenance_in_non_git_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let id = KernelIdentifier::new(
            tmp.path(),
            "relu".to_owned(),
            Some("x".to_owned()),
            None,
            None,
        );

        // No overrides supplied and the temp dir is not a git repository, so
        // no kernel provenance is recorded. The kernel-builder provenance is
        // filled in later from the compile-time default in `write_metadata`.
        assert!(id.git_info().is_none());
        assert!(id.kernel_builder().is_none());
    }
}
