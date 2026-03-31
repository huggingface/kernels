use std::path::PathBuf;

use eyre::Result;

use crate::list_variants::variants;
use crate::nix::{Flake, Nix, NixSubcommand};
use crate::pyproject::write_card;
use crate::util::{check_or_infer_kernel_dir, parse_build};

pub fn run_build(
    kernel_dir: Option<PathBuf>,
    max_jobs: Option<u32>,
    cores: Option<u32>,
    print_build_logs: bool,
    variant: Option<String>,
    target: &str,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;

    if let Ok(build) = parse_build(&kernel_dir) {
        if let Err(e) = write_card(&build, &kernel_dir) {
            eprintln!("Warning: cannot generate CARD.md: {e}");
        }
    }

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

    let attribute = match variant {
        Some(ref v) => Some(format!("redistributable.{v}")),
        None => Some(target.to_owned()),
    };

    let mut nix = Nix::new();
    if let Some(jobs) = max_jobs {
        nix = nix.max_jobs(jobs);
    }
    if let Some(c) = cores {
        nix = nix.cores(c);
    }
    nix = nix.print_build_logs(print_build_logs);

    nix.run(NixSubcommand::Run {
        flake: &flake,
        attribute,
    })
}
