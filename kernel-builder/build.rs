use std::process::Command;

fn main() {
    minijinja_embed::embed_templates!("src/pyproject/templates");
    emit_git_info();
}

fn emit_git_info() {
    let sha = env_var("KERNEL_BUILDER_GIT_SHA").or_else(|| git_output(&["rev-parse", "HEAD"]));
    if let Some(sha) = sha {
        println!("cargo:rustc-env=KERNEL_BUILDER_GIT_SHA={sha}");
    }

    let dirty = match env_var("KERNEL_BUILDER_GIT_DIRTY") {
        Some(value) => is_truthy(&value),
        // Only consider tracked files; untracked files (e.g. generated build
        // artifacts) should not mark the build as dirty.
        None => git_output(&["status", "--porcelain", "--untracked-files=no"])
            .map(|out| !out.is_empty())
            .unwrap_or(false),
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

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|s| !s.is_empty())
}

fn is_truthy(value: &str) -> bool {
    value == "1" || value.eq_ignore_ascii_case("true")
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
