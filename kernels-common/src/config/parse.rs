use std::fs::File;
use std::io::Read;
use std::path::Path;

use eyre::{Context, Result, bail};

use super::{Build, BuildCompat, CurrentConfig};

pub(crate) fn parse_and_validate(kernel_dir: impl AsRef<Path>) -> Result<CurrentConfig> {
    // v4 and v5 are upgraded to v6 in memory so that kernel-builder commands keep
    // working, but the Nix builder only accepts v6, so warn to run `update-build`.
    // Older editions must be migrated explicitly with `update-build`.
    match parse_and_validate_compat(kernel_dir)? {
        BuildCompat::V6(build) => Ok(build),
        BuildCompat::V5(build) => {
            warn_in_memory_upgrade("edition 5");
            Ok(Build::from(build).into())
        }
        BuildCompat::V4(build) => {
            warn_in_memory_upgrade("legacy v4");
            Ok(Build::from(build).into())
        }
        BuildCompat::V3(_) => bail!(
            "build.toml uses an unsupported legacy format; migrate it with \
             `kernel-builder update-build`"
        ),
    }
}

fn warn_in_memory_upgrade(format: &str) {
    eprintln!(
        "⚠️  build.toml uses the {format} format; upgrading to edition 6 in memory. \
         Run `kernel-builder update-build` to persist the upgrade, the Nix builder \
         requires edition 6."
    );
}

pub(crate) fn parse_and_validate_compat(kernel_dir: impl AsRef<Path>) -> Result<BuildCompat> {
    let build_toml = kernel_dir.as_ref().join("build.toml");
    let mut toml_data = String::new();
    File::open(&build_toml)
        .wrap_err_with(|| format!("Cannot open {} for reading", build_toml.to_string_lossy()))?
        .read_to_string(&mut toml_data)
        .wrap_err_with(|| format!("Cannot read from {}", build_toml.to_string_lossy()))?;

    let config: BuildCompat = toml::from_str(&toml_data)
        .wrap_err_with(|| format!("Cannot parse TOML in {}", build_toml.to_string_lossy()))?;

    Ok(config)
}
