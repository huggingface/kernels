//! Development shell CI command implementations.

use std::path::PathBuf;

use eyre::Result;

use crate::nix::{Flake, Nix, NixSubcommand};
use crate::util::check_or_infer_kernel_dir;

/// Run a Nix development shell.
fn run_develop(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    attribute: Option<String>,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let flake = Flake::from_path(kernel_dir)?;

    let mut nix = Nix::new();

    if let Some(max_jobs) = max_jobs {
        nix = nix.max_jobs(max_jobs);
    }

    if let Some(cores) = cores {
        nix = nix.cores(cores);
    }

    nix = nix.print_build_logs(print_build_logs);

    nix.run(NixSubcommand::Develop { flake, attribute })
}

/// Run a kernel development shell.
pub fn devshell(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
) -> Result<()> {
    run_develop(kernel_dir, max_jobs, cores, print_build_logs, None)
}

/// Run a kernel test shell.
pub fn testshell(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
) -> Result<()> {
    run_develop(
        kernel_dir,
        max_jobs,
        cores,
        print_build_logs,
        Some("test".to_string()),
    )
}
