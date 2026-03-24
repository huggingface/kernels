use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

use clap::Args;
use eyre::{bail, Context, Result};
use huggingface_hub::{
    AddSource, CommitOperation, CreateBranchParams, CreateCommitParams, CreateRepoParams,
    ListRepoFilesParams, ListRepoRefsParams, RepoType,
};
use walkdir::WalkDir;

use crate::{
    hf,
    pyproject::parse_metadata,
    util::{check_or_infer_kernel_dir, parse_build},
};

const MAIN_BRANCH: &str = "main";
const BUILD_COMMIT_BATCH_SIZE: usize = 1_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum RepoTypeArg {
    #[default]
    Model,
    Kernel,
}

impl From<RepoTypeArg> for RepoType {
    fn from(arg: RepoTypeArg) -> Self {
        match arg {
            RepoTypeArg::Model => RepoType::Model,
            RepoTypeArg::Kernel => RepoType::Kernel,
        }
    }
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

    /// TODO: remove when we fully move over to kernel repos
    /// Repository type on Hugging Face Hub (`model` or `kernel`).
    #[arg(long, value_enum, default_value_t = RepoTypeArg::Model)]
    pub repo_type: RepoTypeArg,
}

pub fn run_upload(args: UploadArgs) -> Result<()> {
    let api = hf::api()?;
    let repo_type: RepoType = args.repo_type.into();
    let kernel_dir = check_or_infer_kernel_dir(args.kernel_dir)?;
    let kernel_dir = fs::canonicalize(&kernel_dir)
        .wrap_err_with(|| format!("Cannot resolve kernel directory `{}`", kernel_dir.display()))?;

    let build = parse_build(&kernel_dir)?;
    let arg_repo_id = match args.repo_id {
        Some(id) => id,
        None => build
            .repo_id()
            .ok_or_else(|| {
                eyre::eyre!("No `general.hub.repo-id` in build.toml. Use --repo-id to specify it.")
            })?
            .to_owned(),
    };

    let (build_dir, variants) = discover_variants(&kernel_dir)?;
    eprintln!(
        "Found {} build variant(s) in {}",
        variants.len(),
        build_dir.display()
    );

    let version_branch = args
        .branch
        .map_or_else(|| detect_branch_from_metadata(&variants), |b| Ok(Some(b)))?;

    let params = CreateRepoParams::builder()
        .repo_id(&arg_repo_id)
        .repo_type(repo_type)
        .private(args.private)
        .exist_ok(true)
        .build();
    let repo_url = api
        .create_repo(&params)
        .wrap_err("Cannot create repository")?;
    // Extract repo_id from URL, stripping "kernels/" prefix for kernel repos
    let repo_id = repo_url
        .url
        .trim_end_matches('/')
        .strip_prefix("https://huggingface.co/")
        .map(|s| s.strip_prefix("kernels/").unwrap_or(s))
        .unwrap_or(&arg_repo_id)
        .to_owned();

    let is_new_version_branch = if let Some(ref branch) = version_branch {
        let refs_params = ListRepoRefsParams::builder()
            .repo_id(&repo_id)
            .repo_type(repo_type)
            .build();
        let refs = api
            .list_repo_refs(&refs_params)
            .wrap_err("Cannot list repository refs")?;
        let exists = refs.branches.iter().any(|r| r.name == *branch);

        if !exists {
            let params = CreateBranchParams::builder()
                .repo_id(&repo_id)
                .branch(branch)
                .repo_type(repo_type)
                .build();
            api.create_branch(&params)
                .wrap_err_with(|| format!("Cannot create branch `{branch}`"))?;
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

    collect_readme_commit_ops(
        &kernel_dir,
        operations_by_branch
            .entry(MAIN_BRANCH.to_owned())
            .or_default(),
    );

    if let Some(ref branch) = version_branch {
        let params = ListRepoFilesParams {
            repo_id: repo_id.to_owned(),
            revision: Some(branch.clone()),
            repo_type: Some(repo_type),
        };
        let version_existing_files: Vec<String> = api.list_repo_files(&params).unwrap_or_default();

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

        eprintln!(
            "Uploading {} operations to branch `{}`...",
            operations.len(),
            branch
        );

        let batch_count = operations.len().div_ceil(BUILD_COMMIT_BATCH_SIZE);
        if batch_count > 1 {
            eprintln!(
                "Uploading in {} commits ({} operations).",
                batch_count,
                operations.len()
            );
        }

        for (batch_index, chunk) in operations.chunks(BUILD_COMMIT_BATCH_SIZE).enumerate() {
            let commit_message = if batch_count > 1 {
                format!(
                    "Uploaded using `kernel-builder` (batch {}/{batch_count}).",
                    batch_index + 1
                )
            } else {
                "Uploaded using `kernel-builder`.".to_owned()
            };

            let params = CreateCommitParams {
                repo_id: repo_id.to_owned(),
                operations: chunk.to_vec(),
                commit_message,
                commit_description: None,
                repo_type: Some(repo_type),
                revision: Some(branch.clone()),
                create_pr: None,
                parent_commit: None,
            };
            api.create_commit(&params)
                .wrap_err_with(|| format!("Cannot create commit on branch `{branch}`"))?;

            if batch_count > 1 {
                eprintln!("  Uploaded batch {}/{batch_count}.", batch_index + 1);
            }
        }
    }

    let total_ops: usize = operations_by_branch.values().map(|v| v.len()).sum();
    if total_ops == 0 {
        eprintln!("No changes to upload.");
    } else {
        let type_prefix = match repo_type {
            RepoType::Kernel => "kernels/",
            _ => "",
        };
        let tree_path = version_branch
            .as_ref()
            .map_or(String::new(), |b| format!("/tree/{b}"));
        println!("Kernel uploaded: https://hf.co/{type_prefix}{repo_id}{tree_path}");
    }

    Ok(())
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
                .is_some_and(|n| n.starts_with("benchmark"))
        {
            operations.push(CommitOperation::Delete {
                path_in_repo: file.clone(),
            });
        }
    }

    Ok(())
}

/// Collect README commit operation: upload build/CARD.md as README.md.
fn collect_readme_commit_ops(kernel_dir: &Path, operations: &mut Vec<CommitOperation>) {
    let card_path = kernel_dir.join("build").join("CARD.md");
    if card_path.exists() {
        operations.push(CommitOperation::Add {
            path_in_repo: "README.md".to_owned(),
            source: AddSource::File(card_path),
        });
    }
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

/// Discover build variant directories (contain `metadata.json`).
fn discover_variants(kernel_dir: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    for candidate in [kernel_dir.join("build"), kernel_dir.to_path_buf()] {
        if !candidate.is_dir() {
            continue;
        }

        let mut variants: Vec<PathBuf> = fs::read_dir(&candidate)
            .wrap_err_with(|| format!("Cannot read `{}`", candidate.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.join("metadata.json").is_file())
            .collect();

        if !variants.is_empty() {
            variants.sort();
            return Ok((candidate, variants));
        }
    }

    bail!(
        "No build variants found in `{}` or `{}`",
        kernel_dir.join("build").display(),
        kernel_dir.display()
    );
}

/// Determine the branch name (`v{version}`) from variant metadata.
fn detect_branch_from_metadata(variants: &[PathBuf]) -> Result<Option<String>> {
    let mut versions: HashSet<Option<usize>> = HashSet::new();

    for variant in variants {
        let metadata = parse_metadata(variant.join("metadata.json"))?;
        versions.insert(metadata.version);
    }

    if versions.len() > 1 {
        let strs: Vec<_> = versions
            .iter()
            .map(|v| v.map_or("none".into(), |n| n.to_string()))
            .collect();
        bail!(
            "Found multiple versions in build variants: {}",
            strs.join(", ")
        );
    }

    Ok(versions
        .into_iter()
        .next()
        .flatten()
        .map(|v| format!("v{v}")))
}

/// Recursively walk a directory and return all file paths.
fn walk_files(dir: &Path) -> impl Iterator<Item = PathBuf> {
    WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
}
