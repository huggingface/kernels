use std::path::PathBuf;

use clap::Args;
use eyre::{bail, Context, Result};

/// Common arguments shared by all build commands.
#[derive(Debug, Args)]
pub struct CommonBuildArgs {
    /// Directory of the kernel project (defaults to current directory).
    #[arg(value_name = "KERNEL_DIR", default_value = ".")]
    pub kernel_dir: PathBuf,

    /// Maximum number of Nix build jobs.
    #[arg(long, default_value = "4")]
    pub max_jobs: u32,

    /// Number of CPU cores per build job.
    #[arg(long, default_value = "4")]
    pub cores: u32,
}

pub fn run_build(args: CommonBuildArgs, target: &str) -> Result<()> {
    let flake_ref = format!(".#{target}");

    let mut cmd = std::process::Command::new("nix");
    cmd.args([
        "run",
        "-L",
        "--max-jobs",
        &args.max_jobs.to_string(),
        "--cores",
        &args.cores.to_string(),
        &flake_ref,
    ]);
    cmd.current_dir(&args.kernel_dir);

    let status = cmd
        .status()
        .wrap_err("Cannot run `nix`. Is Nix installed?")?;

    if !status.success() {
        bail!("Build failed with exit code {}", status.code().unwrap_or(1));
    }
    Ok(())
}
