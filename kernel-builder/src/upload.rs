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
use regex::Regex;

use crate::hf;

const BUILD_COMMIT_BATCH_SIZE: usize = 1_000;

#[derive(Debug, Args)]
pub struct UploadArgs {
    /// Directory of the kernel build (defaults to current directory).
    #[arg(value_name = "KERNEL_DIR", default_value = ".")]
    pub kernel_dir: PathBuf,

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
}

pub fn run_upload(args: UploadArgs) -> Result<()> {
    let rt = hf::runtime()?;
    let api = hf::api()?;
    let kernel_dir = fs::canonicalize(&args.kernel_dir).wrap_err_with(|| {
        format!(
            "Cannot resolve kernel directory `{}`",
            args.kernel_dir.display()
        )
    })?;

    // Resolve repo_id: explicit --repo-id flag, or read from build.toml.
    let arg_repo_id = match args.repo_id {
        Some(id) => id,
        None => read_repo_id_from_build_toml(&kernel_dir)?,
    };

    // Discover build variants.
    let (build_dir, variants) = discover_variants(&kernel_dir)?;
    eprintln!(
        "Found {} build variant(s) in {}",
        variants.len(),
        build_dir.display()
    );

    // Determine branch from metadata version if not specified.
    let branch = args
        .branch
        .map_or_else(|| detect_branch_from_metadata(&variants), |b| Ok(Some(b)))?;

    // Create repo (or get existing).
    let repo_id = rt.block_on(async {
        let params = CreateRepoParams::builder()
            .repo_id(&arg_repo_id)
            .private(args.private)
            .exist_ok(true)
            .build();
        let repo_url = api
            .create_repo(&params)
            .await
            .wrap_err("Cannot create repository")?;
        // Extract repo_id from the URL (format: https://huggingface.co/{repo_id}).
        let id = repo_url
            .url
            .trim_end_matches('/')
            .strip_prefix("https://huggingface.co/")
            .unwrap_or(&arg_repo_id)
            .to_owned();
        Ok::<_, eyre::Report>(id)
    })?;

    // Create branch if needed.
    let is_new_branch = if let Some(ref branch) = branch {
        let new = rt.block_on(async {
            let refs_params = ListRepoRefsParams::builder().repo_id(&repo_id).build();
            let refs = api
                .list_repo_refs(&refs_params)
                .await
                .wrap_err("Cannot list repository refs")?;
            let exists = refs.branches.iter().any(|r| r.name == *branch);

            if !exists {
                let params = CreateBranchParams::builder()
                    .repo_id(&repo_id)
                    .branch(branch)
                    .build();
                api.create_branch(&params)
                    .await
                    .wrap_err_with(|| format!("Cannot create branch `{branch}`"))?;
            }
            Ok::<_, eyre::Report>(!exists)
        })?;
        eprintln!("Using branch `{branch}`{}", if new { " (new)" } else { "" });
        new
    } else {
        false
    };

    // List existing repo files so we can compute deletions.
    let existing_files: Vec<String> = rt.block_on(async {
        let params = ListRepoFilesParams {
            repo_id: repo_id.to_owned(),
            revision: branch.clone(),
            repo_type: Some(RepoType::Model),
        };
        api.list_repo_files(&params)
            .await
            .wrap_err("Cannot list repository files")
    })?;

    // Collect all operations: benchmarks, README, and build artifacts.
    let mut operations: Vec<CommitOperation> = Vec::new();

    collect_benchmark_ops(&kernel_dir, &existing_files, is_new_branch, &mut operations)?;
    collect_readme_ops(&kernel_dir, &mut operations);
    collect_build_ops(
        &build_dir,
        &variants,
        &existing_files,
        is_new_branch,
        &mut operations,
    )?;

    if operations.is_empty() {
        eprintln!("No changes to upload.");
        return Ok(());
    }

    eprintln!("Uploading {} operations...", operations.len());

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

        rt.block_on(async {
            let params = CreateCommitParams {
                repo_id: repo_id.to_owned(),
                operations: chunk.to_vec(),
                commit_message,
                commit_description: None,
                repo_type: Some(RepoType::Model),
                revision: branch.clone(),
                create_pr: None,
                parent_commit: None,
            };
            api.create_commit(&params)
                .await
                .wrap_err("Cannot create commit")
        })?;

        if batch_count > 1 {
            eprintln!("  Uploaded batch {}/{batch_count}.", batch_index + 1);
        }
    }

    println!("Kernel upload successful. Find the kernel at: https://hf.co/{repo_id}");
    Ok(())
}

// Collect benchmark file operations: add matching files, delete stale ones.
fn collect_benchmark_ops(
    kernel_dir: &Path,
    existing_files: &[String],
    is_new_branch: bool,
    operations: &mut Vec<CommitOperation>,
) -> Result<()> {
    let benchmarks_dir = kernel_dir.join("benchmarks");
    if !benchmarks_dir.is_dir() {
        return Ok(());
    }

    // Add local benchmark files.
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

    // Delete stale benchmark files from the repo.
    for file in existing_files {
        if !file.starts_with("benchmarks/") || added.contains(file) {
            continue;
        }
        // On new branches, delete everything under benchmarks/.
        // On existing branches, only delete benchmark*.py files.
        if is_new_branch
            || file
                .split('/')
                .last()
                .is_some_and(|n| n.starts_with("benchmark"))
        {
            operations.push(CommitOperation::Delete {
                path_in_repo: file.clone(),
            });
        }
    }

    Ok(())
}

// Collect README operation: upload build/CARD.md as README.md.
fn collect_readme_ops(kernel_dir: &Path, operations: &mut Vec<CommitOperation>) {
    let card_path = kernel_dir.join("build").join("CARD.md");
    if card_path.exists() {
        operations.push(CommitOperation::Add {
            path_in_repo: "README.md".to_owned(),
            source: AddSource::File(card_path),
        });
    }
}

// Collect build artifact operations: add variant files, delete stale ones.
fn collect_build_ops(
    build_dir: &Path,
    variants: &[PathBuf],
    existing_files: &[String],
    is_new_branch: bool,
    operations: &mut Vec<CommitOperation>,
) -> Result<()> {
    // Collect local files to repo paths.
    let mut repo_paths: BTreeMap<String, PathBuf> = BTreeMap::new();
    for variant in variants {
        for entry in walkdir(variant)? {
            if entry.is_file() {
                let relative = entry
                    .strip_prefix(build_dir)
                    .wrap_err("Cannot compute relative path")?;
                let repo_path = format!("build/{}", relative.to_string_lossy().replace('\\', "/"));
                repo_paths.insert(repo_path, entry);
            }
        }
    }

    // Determine which prefixes to delete.
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

    // Delete stale files under the relevant prefixes.
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

    // Add build files.
    for (repo_path, local_path) in repo_paths {
        operations.push(CommitOperation::Add {
            path_in_repo: repo_path,
            source: AddSource::File(local_path),
        });
    }

    Ok(())
}

// Read `general.hub.repo-id` from `build.toml` in the kernel directory.
fn read_repo_id_from_build_toml(kernel_dir: &Path) -> Result<String> {
    let build_toml_path = kernel_dir.join("build.toml");
    let content = fs::read_to_string(&build_toml_path)
        .wrap_err_with(|| format!("Cannot read `{}`", build_toml_path.display()))?;
    let parsed: toml::Value = toml::from_str(&content)
        .wrap_err_with(|| format!("Cannot parse `{}`", build_toml_path.display()))?;

    parsed
        .get("general")
        .and_then(|g| g.get("hub"))
        .and_then(|h| h.get("repo-id"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .ok_or_else(|| {
            eyre::eyre!(
                "No `general.hub.repo-id` in `{}`. Use --repo-id to specify it.",
                build_toml_path.display()
            )
        })
}

// Discover build variant directories. Checks `kernel_dir/build/` first, then `kernel_dir/`.
fn discover_variants(kernel_dir: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    let variant_re =
        Regex::new(r"^(torch\d+\d+|torch-(cpu|cuda|metal|neuron|rocm|xpu)|tvm-ffi\d+\d+)")
            .expect("valid regex");

    for candidate in [kernel_dir.join("build"), kernel_dir.to_path_buf()] {
        if !candidate.is_dir() {
            continue;
        }
        let mut variants: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&candidate)
            .wrap_err_with(|| format!("Cannot read `{}`", candidate.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else {
                continue;
            };
            if variant_re.is_match(name_str) && path.join("metadata.json").is_file() {
                variants.push(path);
            }
        }
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

// Read metadata.json from each variant to determine the branch name (`v{version}`).
fn detect_branch_from_metadata(variants: &[PathBuf]) -> Result<Option<String>> {
    let mut versions: HashSet<Option<u64>> = HashSet::new();

    for variant in variants {
        let metadata_path = variant.join("metadata.json");
        let data = fs::read_to_string(&metadata_path)
            .wrap_err_with(|| format!("Cannot read `{}`", metadata_path.display()))?;
        let parsed: serde_json::Value = serde_json::from_str(&data)
            .wrap_err_with(|| format!("Cannot parse `{}`", metadata_path.display()))?;
        let version = parsed.get("version").and_then(|v| v.as_u64());
        versions.insert(version);
    }

    if versions.len() > 1 {
        let version_strs: Vec<String> = versions
            .iter()
            .map(|v| match v {
                Some(n) => n.to_string(),
                None => "none".to_owned(),
            })
            .collect();
        bail!(
            "Found multiple versions in build variants: {}",
            version_strs.join(", ")
        );
    }

    match versions.into_iter().next() {
        Some(Some(version)) => Ok(Some(format!("v{version}"))),
        _ => Ok(None),
    }
}

// Recursively walk a directory and return all file paths.
fn walkdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    walk_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk_recursive(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in
        fs::read_dir(dir).wrap_err_with(|| format!("Cannot read directory `{}`", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_recursive(&path, files)?;
        } else {
            files.push(path);
        }
    }
    Ok(())
}
