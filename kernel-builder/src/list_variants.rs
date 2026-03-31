use std::path::PathBuf;

use eyre::Result;

use crate::nix::{Flake, Nix, NixSubcommand};
use crate::util::check_or_infer_kernel_dir;

/// Get the list of variants for a flake.
pub fn variants(flake: &Flake) -> Result<Vec<String>> {
    let nix = Nix::new().json(true);
    let output = nix.output(NixSubcommand::Eval {
        flake,
        attribute: "variants".to_string(),
    })?;

    let variants: Vec<String> = serde_json::from_slice(&output)?;

    Ok(variants)
}

/// Get the list of architecture-specific variants for a flake.
pub fn arch_variants(flake: &Flake) -> Result<Vec<String>> {
    let nix = Nix::new().json(true);
    let output = nix.output(NixSubcommand::Eval {
        flake,
        attribute: "archVariants".to_string(),
    })?;

    let variants: Vec<String> = serde_json::from_slice(&output)?;

    Ok(variants)
}

pub fn list_variants(kernel_dir: Option<PathBuf>, arch: bool) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let flake = Flake::from_path(kernel_dir)?;

    let variant_list = if arch {
        arch_variants(&flake)?
    } else {
        variants(&flake)?
    };

    for variant in &variant_list {
        println!("{}", variant);
    }

    Ok(())
}
