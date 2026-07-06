fn main() {
    minijinja_embed::embed_templates!("src/pyproject/templates");
    emit_git_info();
}

struct GitInfo {
    sha: String,
    dirty: bool,
}

/// Burn the `kernel-builder` git provenance into the binary at compile time.
///
/// The provenance is read from the git repository using `git2` (rather than
/// shelling out to `git`, which is more robust in build sandboxes). When the
/// build happens without a `.git` (e.g. the Nix sandbox), the derivation passes
/// the provenance through the `KERNEL_BUILDER_GIT_SHA` / `KERNEL_BUILDER_GIT_DIRTY`
/// environment variables, which take precedence.
fn emit_git_info() {
    let git = git_info();

    let sha = env_var("KERNEL_BUILDER_GIT_SHA").or_else(|| git.as_ref().map(|g| g.sha.clone()));
    if let Some(sha) = sha {
        println!("cargo:rustc-env=KERNEL_BUILDER_GIT_SHA={sha}");
    }

    let dirty = match env_var("KERNEL_BUILDER_GIT_DIRTY") {
        Some(value) => is_truthy(&value),
        None => git.map(|g| g.dirty).unwrap_or(false),
    };
    println!(
        "cargo:rustc-env=KERNEL_BUILDER_GIT_DIRTY={}",
        if dirty { "1" } else { "0" }
    );

    println!("cargo:rerun-if-env-changed=KERNEL_BUILDER_GIT_SHA");
    println!("cargo:rerun-if-env-changed=KERNEL_BUILDER_GIT_DIRTY");
    println!("cargo:rerun-if-changed=../.git/HEAD");
    println!("cargo:rerun-if-changed=../.git/index");
}

/// Detect the current commit SHA and dirty state via `git2`.
fn git_info() -> Option<GitInfo> {
    let repo = git2::Repository::discover(".").ok()?;
    let head = repo.head().ok()?;
    let commit = head.peel_to_commit().ok()?;
    let sha = commit.id().to_string();

    let mut status_options = git2::StatusOptions::new();
    // Only consider tracked files; untracked files (e.g. generated build
    // artifacts) should not mark the build as dirty.
    status_options.include_untracked(false);
    status_options.exclude_submodules(true);
    let dirty = !repo.statuses(Some(&mut status_options)).ok()?.is_empty();

    Some(GitInfo { sha, dirty })
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn is_truthy(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}
