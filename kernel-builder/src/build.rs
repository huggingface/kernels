use std::path::PathBuf;

use clap::Args;
use eyre::Result;

use crate::nix::{Flake, Nix, NixSubcommand};
use crate::pyproject::write_card;
use crate::util::{check_or_infer_kernel_dir, parse_build};

/// Common arguments shared by all build commands.
#[derive(Debug, Args)]
pub struct CommonBuildArgs {
    /// Directory of the kernel project (defaults to current directory).
    #[arg(value_name = "KERNEL_DIR")]
    pub kernel_dir: Option<PathBuf>,

    /// Maximum number of Nix build jobs.
    #[arg(long, default_value = "4")]
    pub max_jobs: u32,

    /// Number of CPU cores per build job.
    #[arg(long, default_value = "4")]
    pub cores: u32,
}

pub fn run_build(args: CommonBuildArgs, target: &str) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(args.kernel_dir)?;

    if let Ok(build) = parse_build(&kernel_dir) {
        if let Err(e) = write_card(&build, &kernel_dir) {
            eprintln!("Warning: cannot generate CARD.md: {e}");
        }
    }

    let flake = Flake::from_path(kernel_dir)?;

    Nix::new()
        .max_jobs(args.max_jobs)
        .cores(args.cores)
        .run(NixSubcommand::Run {
            flake,
            attribute: Some(target.to_owned()),
        })
}
