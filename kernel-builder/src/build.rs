use std::{fmt::Display, path::PathBuf, str::FromStr};

use clap::Args;
use eyre::{bail, Context, Result};

#[derive(Clone, Debug, Default)]
pub enum BuildTarget {
    #[default]
    BuildAndCopy,
    BuildAndUpload,
    Ci,
    Default,
}

impl Display for BuildTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildTarget::BuildAndCopy => write!(f, "build-and-copy"),
            BuildTarget::BuildAndUpload => write!(f, "build-and-upload"),
            BuildTarget::Ci => write!(f, "ci"),
            BuildTarget::Default => write!(f, "default"),
        }
    }
}

impl FromStr for BuildTarget {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "build-and-copy" => Ok(BuildTarget::BuildAndCopy),
            "build-and-upload" => Ok(BuildTarget::BuildAndUpload),
            "ci" => Ok(BuildTarget::Ci),
            "default" => Ok(BuildTarget::Default),
            _ => Err(format!(
                "unknown target `{s}`, expected one of: build-and-copy, build-and-upload, ci, default"
            )),
        }
    }
}

#[derive(Debug, Args)]
pub struct BuildArgs {
    /// Directory of the kernel project (defaults to current directory).
    #[arg(value_name = "KERNEL_DIR", default_value = ".")]
    pub kernel_dir: PathBuf,

    /// Nix flake target to run.
    #[arg(long, default_value = "build-and-copy")]
    pub target: BuildTarget,

    /// Maximum number of Nix build jobs.
    #[arg(long, default_value = "4")]
    pub max_jobs: u32,

    /// Number of CPU cores per build job.
    #[arg(long, default_value = "4")]
    pub cores: u32,

    /// Additional arguments passed through to `nix run`.
    #[arg(last = true)]
    pub nix_args: Vec<String>,
}

pub fn run_build(args: BuildArgs) -> Result<()> {
    let flake_ref = format!(".#{}", args.target);

    let mut cmd = std::process::Command::new("nix");
    cmd.args([
        "run",
        "-L",
        "--max-jobs",
        &args.max_jobs.to_string(),
        "--cores",
        &args.cores.to_string(),
    ]);
    cmd.args(&args.nix_args);
    cmd.arg(&flake_ref);
    cmd.current_dir(&args.kernel_dir);

    let status = cmd
        .status()
        .wrap_err("Cannot run `nix`. Is Nix installed?")?;

    if !status.success() {
        bail!("Build failed with exit code {}", status.code().unwrap_or(1));
    }
    Ok(())
}
