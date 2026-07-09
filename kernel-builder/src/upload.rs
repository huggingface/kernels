use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
};

use clap::Args;
use eyre::{bail, eyre, Context, Result};
use hf_hub::{
    progress::{Progress, ProgressEvent, ProgressHandler, UploadEvent},
    repository::{AddSource, CommitInfo, CommitOperation},
    HFError, HFRepositorySync, RepoType, RepoTypeKernel, RepoTypeModel,
};
use indicatif::{ProgressBar, ProgressStyle};
use kernels_data::metadata::Metadata;
use walkdir::WalkDir;

use crate::{
    hf::{self, repo_handle},
    util::{check_or_infer_kernel_dir, discover_variants, parse_build},
};

/// Bridges `ProgressHandler` events to an `indicatif::ProgressBar`.
struct IndicatifProgress(ProgressBar);

impl ProgressHandler for IndicatifProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Upload(UploadEvent::Progress { files, .. }) = event {
            let completed = files
                .iter()
                .filter(|f| f.status == hf_hub::progress::FileStatus::Complete)
                .count();
            self.0.set_position(completed as u64);
        }
    }
}

const MAIN_BRANCH: &str = "main";
const BUILD_COMMIT_BATCH_SIZE: usize = 1_000;

const KERNELS_COMMUNITY_URL: &str =
    "https://huggingface.co/spaces/kernels-community/README/discussions/new";

fn kernel_publishing_guidance(repo_id: &str) -> String {
    format!(
        "You do not have permission to publish the kernel `{repo_id}`.

Publishing kernels to the Hugging Face Hub requires approval. To request access,
open a discussion on the `kernels-community` organization:

    {KERNELS_COMMUNITY_URL}

In the discussion, please include:
  - Confirmation that you already have a working, built kernel.
  - The source URI / code for the kernel.
  - The kernel's purpose and where it fits in the ecosystem (which models or
    libraries benefit from it).
  - Benchmark numbers, if available.

Providing this information helps the team review your request and grant access."
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum RepoTypeArg {
    Model,
    #[default]
    Kernel,
}

#[derive(Debug, Args)]
pub struct UploadArgs {
    /// Directory of the kernel build (defaults to current directory).
    #[arg(value_name = "KERNEL_DIR")]
    pub kernel_dir: Option<PathBuf>,

    /// Repository ID on the Hugging Face Hub (e.g. `user/my-kernel`).
    /// Defaults to `general.hub.repo-id` from `build.toml`.
    #[arg(long)]
    pub repo_id: Option<String>,

    /// Upload to a specific branch (defaults to `v{version}` from metadata).
    #[arg(long)]
    pub branch: Option<String>,

    /// Create the repository as private.
    #[arg(long)]
    pub private: bool,

    /// Repository type on Hugging Face Hub (`kernel` by default, or `model` for legacy repos).
    #[arg(long, value_enum, default_value_t = RepoTypeArg::Kernel)]
    pub repo_type: RepoTypeArg,

    /// Open a pull request with the changes rather than committing them
    /// directly (does not require write access to the repository).
    #[arg(long)]
    pub create_pr: bool,

    /// Suppress progress output.
    #[arg(long, short)]
    pub quiet: bool,
}

/// Get repository and branch from the given arguments, or fallback to
/// reading `build.toml` and/or metadata otherwise.
fn get_repo_and_branch(
    kernel_dir: &Path,
    repo_id: Option<String>,
    branch: Option<String>,
    variants: &[PathBuf],
) -> Result<(String, Option<String>)> {
    let build = parse_build(kernel_dir);

    let build_branch = build
        .as_ref()
        .ok()
        .and_then(|b| b.branch().map(ToOwned::to_owned));
    let arg_branch = branch.or(build_branch);

    let resolved_repo_id = match repo_id {
        Some(id) => id,
        None => build
            .context("--repo-id is not provided and cannot parse build.toml.")?
            .repo_id()
            .ok_or_else(|| {
                eyre::eyre!("No `general.hub.repo-id` in build.toml. Use --repo-id to specify it.")
            })?
            .to_owned(),
    };

    let version_branch =
        arg_branch.map_or_else(|| detect_branch_from_metadata(variants), |b| Ok(Some(b)))?;

    Ok((resolved_repo_id, version_branch))
}

pub fn run_upload(args: UploadArgs) -> Result<()> {
    match args.repo_type {
        RepoTypeArg::Model => run_upload_typed::<RepoTypeModel>(args),
        RepoTypeArg::Kernel => run_upload_typed::<RepoTypeKernel>(args),
    }
}

fn run_upload_typed<T: RepoType>(args: UploadArgs) -> Result<()> {
    let api = hf::api()?;
    let kernel_dir = check_or_infer_kernel_dir(args.kernel_dir)?;
    let kernel_dir = fs::canonicalize(&kernel_dir)
        .wrap_err_with(|| format!("Cannot resolve kernel directory `{}`", kernel_dir.display()))?;

    let (build_dir, variants) = discover_variants(&kernel_dir)?;
    eprintln!(
        "Found {} build variant(s) in {}",
        variants.len(),
        build_dir.display()
    );

    let (repo_id, branch) = get_repo_and_branch(&kernel_dir, args.repo_id, args.branch, &variants)?;

    // With --create-pr, write access is not required and we must not create a
    // new repository: the repository only needs to already exist for the Hub to
    // open a pull request against it. Skip repository creation entirely and use
    // the provided repo id as-is.
    let repo_id = if args.create_pr {
        repo_id
    } else {
        let repo_url = match api
            .create_repository()
            .repo_id(&repo_id)
            .repo_type(T::default())
            .private(args.private)
            .exist_ok(true)
            .send()
        {
            Ok(url) => url,
            Err(err @ HFError::Forbidden { .. }) => {
                if hf::whoami_username().is_err() {
                    return Err(err).wrap_err("Cannot create repository");
                }
                bail!(kernel_publishing_guidance(&repo_id));
            }
            Err(err) => return Err(err).wrap_err("Cannot create repository"),
        };
        // Extract repo_id from URL, stripping "kernels/" prefix for kernel repos.
        repo_url
            .url
            .trim_end_matches('/')
            .strip_prefix("https://huggingface.co/")
            .map(|s| s.strip_prefix("kernels/").unwrap_or(s).to_owned())
            .unwrap_or(repo_id)
    };

    let repo = repo_handle::<T>(&api, &repo_id);

    let is_new_version_branch = if let Some(ref branch) = branch {
        let refs = repo
            .list_refs()
            .send()
            .wrap_err("Cannot list repository refs")?;
        let exists = refs.branches.iter().any(|r| r.name == *branch);

        if !exists {
            repo.create_branch()
                .branch(branch)
                .send()
                .wrap_err_with(|| {
                    if args.create_pr {
                        format!(
                            "Pull requests can only target an existing branch. Ask a \
                             maintainer of `{repo_id}` to create the branch `{branch}` first."
                        )
                    } else {
                        format!("Cannot create branch `{branch}`")
                    }
                })?;
        }
        eprintln!(
            "Using branch `{branch}`{}",
            if !exists { " (new)" } else { "" }
        );
        !exists
    } else {
        false
    };

    // README goes to main branch, build artifacts go to version branch.
    let mut operations_by_branch: BTreeMap<String, Vec<CommitOperation>> = BTreeMap::new();
    let mut pr_urls: Vec<String> = Vec::new();

    collect_readme_commit_ops(
        &build_dir,
        operations_by_branch
            .entry(MAIN_BRANCH.to_owned())
            .or_default(),
    );

    if let Some(ref branch) = branch {
        let version_existing_files: Vec<String> = repo
            .list_tree()
            .revision(branch.clone())
            .recursive(true)
            .send()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|entry| match entry {
                hf_hub::repository::RepoTreeEntry::File { path, .. } => Some(path),
                hf_hub::repository::RepoTreeEntry::Directory { .. } => None,
            })
            .collect();

        let version_ops = operations_by_branch.entry(branch.clone()).or_default();

        collect_benchmark_commit_ops(
            &kernel_dir,
            &version_existing_files,
            is_new_version_branch,
            version_ops,
        )?;
        collect_build_commit_ops(
            &build_dir,
            &variants,
            &version_existing_files,
            is_new_version_branch,
            version_ops,
        )?;
    }

    for (branch, operations) in &operations_by_branch {
        if operations.is_empty() {
            continue;
        }

        let batch_count = operations.len().div_ceil(BUILD_COMMIT_BATCH_SIZE);
        let progress = if args.quiet {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(operations.len() as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "Uploading to `{msg}` [{bar:40.cyan/blue}] {pos}/{len} files",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            pb.set_message(branch.clone());
            pb
        };

        let progress_handler: Option<Progress> = if args.quiet {
            None
        } else {
            let pb = progress.clone();
            Some(Progress::new(IndicatifProgress(pb)))
        };

        // With --create-pr, the first batch opens a pull request against the
        // branch; follow-up batches are pushed to the pull request ref so that
        // one upload always results in a single pull request per branch.
        let mut pr_revision: Option<String> = None;

        for (batch_index, chunk) in operations.chunks(BUILD_COMMIT_BATCH_SIZE).enumerate() {
            let commit_message = if batch_count > 1 {
                format!(
                    "Uploaded using `kernel-builder` (batch {}/{batch_count}).",
                    batch_index + 1
                )
            } else {
                "Uploaded using `kernel-builder`.".to_owned()
            };

            let open_pr = args.create_pr && pr_revision.is_none();
            let revision = pr_revision.clone().unwrap_or_else(|| branch.clone());

            let commit_info = repo
                .create_commit()
                .operations(chunk.to_vec())
                .commit_message(&commit_message)
                .revision(revision)
                .create_pr(open_pr)
                .maybe_progress(progress_handler.clone())
                .send()
                .wrap_err_with(|| format!("Cannot create commit on branch `{branch}`"))?;

            if open_pr {
                let (pr_ref, pr_url) =
                    resolve_pr(&repo, &repo_id, &commit_info).wrap_err_with(|| {
                        format!("Cannot determine the pull request opened for branch `{branch}`")
                    })?;
                pr_revision = Some(pr_ref);
                pr_urls.push(pr_url);
            }
        }

        progress.finish_with_message(if args.create_pr {
            format!("Opened pull request against `{branch}`")
        } else {
            format!("Uploaded to `{branch}`")
        });
    }

    let total_ops: usize = operations_by_branch.values().map(|v| v.len()).sum();
    if total_ops == 0 {
        eprintln!("No changes to upload.");
    } else if args.create_pr {
        for url in &pr_urls {
            println!("Pull request created: {url}");
        }
    } else {
        let type_prefix = T::default().url_prefix();
        let tree_path = branch
            .as_ref()
            .map_or(String::new(), |b| format!("/tree/{b}"));
        println!("Kernel uploaded: https://hf.co/{type_prefix}{repo_id}{tree_path}");
    }

    Ok(())
}

/// Resolve the ref and URL of the pull request opened by a `create_pr` commit.
///
/// The Hub reports the pull request as `pullRequestUrl` in the commit
/// response, which hf-hub 1.0.0-rc.0 does not deserialize into `CommitInfo`
/// (it expects `prUrl`/`prNum`), so fall back to matching the commit OID
/// against the repository's pull-request refs.
fn resolve_pr<T: RepoType>(
    repo: &HFRepositorySync<T>,
    repo_id: &str,
    commit_info: &CommitInfo,
) -> Result<(String, String)> {
    let pr_url_for = |num: &str| {
        format!(
            "https://hf.co/{}{repo_id}/discussions/{num}",
            T::default().url_prefix()
        )
    };

    if let Some(num) = commit_info.pr_num {
        let url = commit_info
            .pr_url
            .clone()
            .unwrap_or_else(|| pr_url_for(&num.to_string()));
        return Ok((format!("refs/pr/{num}"), url));
    }

    let oid = commit_info
        .commit_oid
        .as_deref()
        .ok_or_else(|| eyre!("The Hub returned neither a pull request nor a commit OID"))?;
    let refs = repo
        .list_refs()
        .include_pull_requests(true)
        .send()
        .wrap_err("Cannot list repository refs")?;
    let pr = refs
        .pull_requests
        .iter()
        .find(|r| r.target_commit == oid)
        .ok_or_else(|| eyre!("No pull-request ref points at the created commit `{oid}`"))?;
    let num = pr
        .git_ref
        .rsplit('/')
        .next()
        .ok_or_else(|| eyre!("Malformed pull-request ref `{}`", pr.git_ref))?;

    Ok((pr.git_ref.clone(), pr_url_for(num)))
}

/// Collect benchmark file commit operations: add matching files, delete stale ones.
fn collect_benchmark_commit_ops(
    kernel_dir: &Path,
    existing_files: &[String],
    is_new_branch: bool,
    operations: &mut Vec<CommitOperation>,
) -> Result<()> {
    let benchmarks_dir = kernel_dir.join("benchmarks");
    if !benchmarks_dir.is_dir() {
        return Ok(());
    }

    let mut added: HashSet<String> = HashSet::new();
    for entry in fs::read_dir(&benchmarks_dir)
        .wrap_err_with(|| format!("Cannot read `{}`", benchmarks_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("benchmark") || !name.ends_with(".py") {
            continue;
        }
        let repo_path = format!("benchmarks/{name}");
        added.insert(repo_path.clone());
        operations.push(CommitOperation::Add {
            path_in_repo: repo_path,
            source: AddSource::File(path),
        });
    }

    for file in existing_files {
        if !file.starts_with("benchmarks/") || added.contains(file) {
            continue;
        }
        // On new branches delete everything; on existing branches only delete benchmark*.py.
        if is_new_branch
            || file
                .split('/')
                .next_back()
                .is_some_and(|n| n.starts_with("benchmark") && n.ends_with(".py"))
        {
            operations.push(CommitOperation::Delete {
                path_in_repo: file.clone(),
            });
        }
    }

    Ok(())
}

/// Collect README commit operation: upload the `CARD.md` as `README.md`.
///
/// The card is only looked for in `build_dir`, i.e. the same directory that
/// holds the build variants (as returned by `discover_variants`). This ensures
/// the card is taken from the same location as the variants rather than an
/// unrelated directory elsewhere in the repository (see issue #659).
fn collect_readme_commit_ops(build_dir: &Path, operations: &mut Vec<CommitOperation>) {
    let card_path = build_dir.join("CARD.md");
    if !card_path.is_file() {
        return;
    }
    operations.push(CommitOperation::Add {
        path_in_repo: "README.md".to_owned(),
        source: AddSource::File(card_path),
    });
}

/// Collect build artifact commit operations: add variant files, delete stale ones.
fn collect_build_commit_ops(
    build_dir: &Path,
    variants: &[PathBuf],
    existing_files: &[String],
    is_new_branch: bool,
    operations: &mut Vec<CommitOperation>,
) -> Result<()> {
    let mut repo_paths: BTreeMap<String, PathBuf> = BTreeMap::new();
    for variant in variants {
        for path in walk_files(variant) {
            let relative = path
                .strip_prefix(build_dir)
                .wrap_err("Cannot compute relative path")?;
            let repo_path = format!("build/{}", relative.to_string_lossy().replace('\\', "/"));
            repo_paths.insert(repo_path, path);
        }
    }

    let variant_prefixes: Vec<String> = variants
        .iter()
        .map(|v| {
            let relative = v.strip_prefix(build_dir).unwrap_or(v);
            format!("build/{}/", relative.to_string_lossy().replace('\\', "/"))
        })
        .collect();

    let delete_prefixes: Vec<&str> = if is_new_branch {
        vec!["build/"]
    } else {
        variant_prefixes.iter().map(|s| s.as_str()).collect()
    };

    for file in existing_files {
        let should_delete = delete_prefixes
            .iter()
            .any(|prefix| file.starts_with(prefix));
        if should_delete && !repo_paths.contains_key(file) {
            operations.push(CommitOperation::Delete {
                path_in_repo: file.clone(),
            });
        }
    }

    for (repo_path, local_path) in repo_paths {
        operations.push(CommitOperation::Add {
            path_in_repo: repo_path,
            source: AddSource::File(local_path),
        });
    }

    Ok(())
}

/// Determine the branch name (`v{version}`) from variant metadata.
fn detect_branch_from_metadata(variants: &[PathBuf]) -> Result<Option<String>> {
    let mut versions: HashSet<usize> = HashSet::new();

    for variant in variants {
        let metadata_path = variant.join("metadata.json");
        let metadata = Metadata::from_reader(BufReader::new(File::open(&metadata_path).context(
            format!(
                "Cannot read metadata from: {}",
                metadata_path.to_string_lossy()
            ),
        )?))?;
        versions.insert(metadata.version);
    }

    if versions.len() > 1 {
        let strs: Vec<_> = versions.iter().map(ToString::to_string).collect();
        bail!(
            "Found multiple versions in build variants: {}",
            strs.join(", ")
        );
    }

    Ok(versions.into_iter().next().map(|v| format!("v{v}")))
}

/// Recursively walk a directory and return all file paths.
fn walk_files(dir: &Path) -> impl Iterator<Item = PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_publishing_guidance() {
        let guidance = kernel_publishing_guidance("my-user/my-kernel");
        assert!(guidance.contains("my-user/my-kernel"));
        assert!(guidance.contains(KERNELS_COMMUNITY_URL));
        assert!(guidance.contains("source URI"));
        assert!(guidance.contains("Benchmark"));
    }

    #[test]
    fn test_collect_readme_commit_ops() {
        let temp_dir = tempfile::tempdir().unwrap();
        let build_dir = temp_dir.path().join("build");

        fs::create_dir_all(&build_dir).unwrap();
        fs::write(build_dir.join("CARD.md"), "# Readme").unwrap();

        let mut operations = vec![];
        collect_readme_commit_ops(&build_dir, &mut operations);

        assert_eq!(operations.len(), 1);
        match &operations[0] {
            CommitOperation::Add {
                path_in_repo,
                source,
            } => {
                assert_eq!(path_in_repo, "README.md");
                match source {
                    AddSource::File(path) => assert_eq!(*path, build_dir.join("CARD.md")),
                    _ => panic!("Expected file source"),
                }
            }
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_collect_readme_commit_ops_no_card() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut operations = vec![];
        collect_readme_commit_ops(temp_dir.path(), &mut operations);
        assert!(operations.is_empty());
    }

    #[test]
    fn test_collect_readme_commit_ops_ignores_card_outside_build_dir() {
        // A card in the kernel directory (or any other directory) must not be
        // picked up when the variants — and thus `build_dir` — live in `build/`.
        // See issue #659.
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();
        let build_dir = kernel_dir.join("build");

        fs::create_dir_all(&build_dir).unwrap();
        // Stray card outside the variants directory.
        fs::write(kernel_dir.join("CARD.md"), "# Stray card").unwrap();

        let mut operations = vec![];
        collect_readme_commit_ops(&build_dir, &mut operations);

        assert!(operations.is_empty());
    }

    #[test]
    fn test_collect_benchmark_commit_ops() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let benchmarks_dir = kernel_dir.join("benchmarks");
        fs::create_dir_all(&benchmarks_dir).unwrap();
        fs::write(benchmarks_dir.join("benchmark.py"), "# benchmark").unwrap();
        fs::write(benchmarks_dir.join("benchmark_v2.py"), "# v2").unwrap();
        fs::write(benchmarks_dir.join("other.py"), "# not a benchmark").unwrap();

        let mut operations = vec![];
        collect_benchmark_commit_ops(kernel_dir, &[], false, &mut operations).unwrap();

        // Should only include benchmark*.py files
        assert_eq!(operations.len(), 2);
    }

    #[test]
    fn test_collect_benchmark_commit_ops_delete_stale() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let benchmarks_dir = kernel_dir.join("benchmarks");
        fs::create_dir_all(&benchmarks_dir).unwrap();
        fs::write(benchmarks_dir.join("benchmark.py"), "# benchmark").unwrap();

        let existing = vec!["benchmarks/benchmark_old.py".to_owned()];
        let mut operations = vec![];
        collect_benchmark_commit_ops(kernel_dir, &existing, false, &mut operations).unwrap();

        // Should add new benchmark and delete stale one
        let add_count = operations
            .iter()
            .filter(|op| matches!(op, CommitOperation::Add { .. }))
            .count();
        let delete_count = operations
            .iter()
            .filter(|op| matches!(op, CommitOperation::Delete { .. }))
            .count();

        assert_eq!(add_count, 1);
        assert_eq!(delete_count, 1);
    }

    #[test]
    fn test_collect_build_commit_ops() {
        let temp_dir = tempfile::tempdir().unwrap();
        let build_dir = temp_dir.path();

        let variant = build_dir.join("torch-cpu");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), "{}").unwrap();
        fs::write(variant.join("kernel.so"), "binary").unwrap();

        let variants = vec![variant];
        let mut operations = vec![];
        collect_build_commit_ops(build_dir, &variants, &[], false, &mut operations).unwrap();

        assert_eq!(operations.len(), 2); // metadata.json + kernel.so
        let paths: Vec<_> = operations
            .iter()
            .filter_map(|op| match op {
                CommitOperation::Add { path_in_repo, .. } => Some(path_in_repo.clone()),
                _ => None,
            })
            .collect();
        assert!(paths.iter().any(|p| p.ends_with("metadata.json")));
        assert!(paths.iter().any(|p| p.ends_with("kernel.so")));
    }

    #[test]
    fn test_collect_build_commit_ops_deletes_stale() {
        let temp_dir = tempfile::tempdir().unwrap();
        let build_dir = temp_dir.path();

        let variant = build_dir.join("torch-cpu");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), "{}").unwrap();

        let existing = vec![
            "build/torch-cpu/stale.py".to_owned(),
            "build/torch-cuda/keep.py".to_owned(), // Different variant, should not delete
        ];
        let variants = vec![variant];
        let mut operations = vec![];
        collect_build_commit_ops(build_dir, &variants, &existing, false, &mut operations).unwrap();

        let delete_paths: Vec<_> = operations
            .iter()
            .filter_map(|op| match op {
                CommitOperation::Delete { path_in_repo } => Some(path_in_repo.clone()),
                _ => None,
            })
            .collect();

        // Only deletes stale files within the same variant
        assert_eq!(delete_paths, vec!["build/torch-cpu/stale.py"]);
    }

    #[test]
    fn test_collect_build_commit_ops_new_branch_deletes_all() {
        let temp_dir = tempfile::tempdir().unwrap();
        let build_dir = temp_dir.path();

        let variant = build_dir.join("torch-cpu");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), "{}").unwrap();

        let existing = vec![
            "build/torch-cpu/stale.py".to_owned(),
            "build/torch-cuda/inherited.py".to_owned(),
        ];
        let variants = vec![variant];
        let mut operations = vec![];
        // is_new_branch = true
        collect_build_commit_ops(build_dir, &variants, &existing, true, &mut operations).unwrap();

        let delete_paths: HashSet<_> = operations
            .iter()
            .filter_map(|op| match op {
                CommitOperation::Delete { path_in_repo } => Some(path_in_repo.clone()),
                _ => None,
            })
            .collect();

        // New branch deletes ALL build/* files not in current upload
        assert!(delete_paths.contains("build/torch-cpu/stale.py"));
        assert!(delete_paths.contains("build/torch-cuda/inherited.py"));
    }

    const METADATA_V3: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 3, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#;
    const METADATA_V0: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 0, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#;

    #[test]
    fn test_detect_branch_from_metadata() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = temp_dir.path().join("variant");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), METADATA_V3).unwrap();

        let variants = vec![variant];
        let branch = detect_branch_from_metadata(&variants).unwrap();
        assert_eq!(branch, Some("v3".to_owned()));
    }

    #[test]
    fn test_detect_branch_from_metadata_v0() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = temp_dir.path().join("variant");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), METADATA_V0).unwrap();

        let variants = vec![variant];
        let branch = detect_branch_from_metadata(&variants).unwrap();
        assert_eq!(branch, Some("v0".to_owned()));
    }

    #[test]
    fn test_detect_branch_from_metadata_mismatched_versions() {
        let temp_dir = tempfile::tempdir().unwrap();
        let v1 = temp_dir.path().join("v1");
        let v2 = temp_dir.path().join("v2");
        fs::create_dir_all(&v1).unwrap();
        fs::create_dir_all(&v2).unwrap();
        fs::write(
            v1.join("metadata.json"),
            r#"{"name": "test-kernel", "version": 1, "id": "k1", "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#,
        )
        .unwrap();
        fs::write(
            v2.join("metadata.json"),
            r#"{"name": "test-kernel", "version": 2, "id": "k2", "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#,
        )
        .unwrap();

        let variants = vec![v1, v2];
        let result = detect_branch_from_metadata(&variants);
        assert!(result.is_err());
    }

    #[test]
    fn test_collect_readme_commit_ops_from_result_symlink() {
        // When variants live in a Nix store output exposed via a `result`
        // symlink, the card next to them must be picked up through the symlink.
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let store_dir = kernel_dir.join("nix-store-output");
        fs::create_dir_all(&store_dir).unwrap();
        fs::write(store_dir.join("CARD.md"), "# Test card").unwrap();

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&store_dir, kernel_dir.join("result")).unwrap();
            let build_dir = kernel_dir.join("result");

            let mut operations = vec![];
            collect_readme_commit_ops(&build_dir, &mut operations);

            assert_eq!(operations.len(), 1);
            match &operations[0] {
                CommitOperation::Add { path_in_repo, .. } => {
                    assert_eq!(path_in_repo, "README.md");
                }
                _ => panic!("Expected Add operation"),
            }
        }
    }

    #[test]
    fn test_branch_from_build_toml() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        fs::write(
            kernel_dir.join("build.toml"),
            r#"[general]
name = "test-kernel"
license = "Apache-2.0"
backends = ["cuda"]
version = 1
edition = 5

[general.hub]
repo-id = "test/kernel"
branch = "custom-branch"

[torch]
"#,
        )
        .unwrap();

        let build_dir = kernel_dir.join("build");
        let variant = build_dir.join("torch-cuda");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), METADATA_V3).unwrap();
        fs::write(variant.join("kernel.so"), "binary").unwrap();

        let variants = vec![variant.clone()];
        let (repo_id, branch) = get_repo_and_branch(kernel_dir, None, None, &variants).unwrap();

        assert_eq!(repo_id, "test/kernel");
        assert_eq!(branch, Some("custom-branch".to_owned()));

        // Verify commit ops are generated - these would be uploaded to the branch above.
        let mut operations = vec![];
        collect_build_commit_ops(&build_dir, &variants, &[], false, &mut operations).unwrap();
        assert!(!operations.is_empty());
    }

    #[test]
    fn test_args_take_priority_over_files() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        fs::write(
            kernel_dir.join("build.toml"),
            r#"[general]
name = "test-kernel"
license = "Apache-2.0"
backends = ["cuda"]

[general.hub]
repo-id = "build-toml/kernel"
branch = "build-toml-branch"
"#,
        )
        .unwrap();

        let build_dir = kernel_dir.join("build");
        let variant = build_dir.join("torch-cuda");
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), METADATA_V3).unwrap();

        let variants = vec![variant];
        let (repo_id, branch) = get_repo_and_branch(
            kernel_dir,
            Some("args/kernel".to_owned()),
            Some("args-branch".to_owned()),
            &variants,
        )
        .unwrap();

        assert_eq!(repo_id, "args/kernel");
        assert_eq!(branch, Some("args-branch".to_owned()));
    }
}
