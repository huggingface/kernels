use std::path::PathBuf;

use eyre::Result;

use crate::list_variants::variants;
use crate::nix::{Flake, Nix, NixSubcommand};
use crate::util::check_or_infer_kernel_dir;

fn prepare_build(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
) -> Result<(Flake, Option<String>, Nix)> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;

    let flake = Flake::from_path(kernel_dir)?;

    if let Some(ref variant) = variant {
        let valid_variants = variants(&flake)?;
        if !valid_variants.contains(variant) {
            eyre::bail!(
                "Unknown variant `{variant}`.\nValid variants are: {}",
                valid_variants.join(", ")
            );
        }
    }

    let mut nix = Nix::new();
    if let Some(jobs) = max_jobs {
        nix = nix.max_jobs(jobs);
    }
    if let Some(c) = cores {
        nix = nix.cores(c);
    }
    nix = nix.print_build_logs(print_build_logs);

    Ok((flake, variant, nix))
}

pub fn run_build(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
) -> Result<()> {
    let (flake, variant, nix) =
        prepare_build(kernel_dir, max_jobs, cores, print_build_logs, variant)?;

    nix.run(NixSubcommand::Build {
        flake: &flake,
        attribute: variant.map(|v| format!("redistributable.{v}")),
    })
}

pub fn run_build_and_copy(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
) -> Result<()> {
    let (flake, _, nix) = prepare_build(kernel_dir, max_jobs, cores, print_build_logs, None)?;

    nix.run(NixSubcommand::Run {
        flake: &flake,
        attribute: Some("build-and-copy".to_owned()),
    })
}
