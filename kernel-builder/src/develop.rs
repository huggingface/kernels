//! Development shell CI command implementations.

use std::path::PathBuf;

use eyre::Result;

use crate::list_variants::arch_variants;
use crate::nix::{Flake, Nix, NixSubcommand};
use crate::util::check_or_infer_kernel_dir;

/// Validate a variant against the arch variants list.
fn validate_arch_variant(flake: &Flake, variant: &str) -> Result<()> {
    let valid_variants = arch_variants(flake)?;
    if !valid_variants.contains(&variant.to_string()) {
        eyre::bail!(
            "Unknown variant `{variant}`.\nValid variants for this architecture are: {}",
            valid_variants.join(", ")
        );
    }
    Ok(())
}

/// Run a Nix development shell.
fn run_develop(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
    shell_prefix: &str,
    default_attribute: Option<String>,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let flake = Flake::from_path(kernel_dir)?;

    if let Some(ref variant) = variant {
        validate_arch_variant(&flake, variant)?;
    }

    let attribute = match variant {
        Some(ref v) => Some(format!("{shell_prefix}.{v}")),
        None => default_attribute,
    };

    let mut nix = Nix::new();

    if let Some(max_jobs) = max_jobs {
        nix = nix.max_jobs(max_jobs);
    }

    if let Some(cores) = cores {
        nix = nix.cores(cores);
    }

    nix = nix.print_build_logs(print_build_logs);

    nix.run(NixSubcommand::Develop {
        flake: &flake,
        attribute,
    })
}

/// Run a kernel development shell.
pub fn devshell(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
) -> Result<()> {
    run_develop(
        kernel_dir,
        max_jobs,
        cores,
        print_build_logs,
        variant,
        "devShells",
        None,
    )
}

/// Run a kernel test shell.
pub fn testshell(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
) -> Result<()> {
    run_develop(
        kernel_dir,
        max_jobs,
        cores,
        print_build_logs,
        variant,
        "testShells",
        Some("test".to_string()),
    )
}
