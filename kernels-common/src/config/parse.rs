use std::fs::File;
use std::io::Read;
use std::path::Path;

use eyre::{Context, Result, bail};

use super::{Build, BuildCompat, CURRENT_EDITION, CurrentConfig};

pub(crate) fn parse_and_validate(kernel_dir: impl AsRef<Path>) -> Result<CurrentConfig> {
    // v4 and v5 are auto-upgraded on load; older editions must be migrated
    // explicitly with `update-build`.
    match parse_and_validate_compat(kernel_dir)? {
        BuildCompat::V6(build) => Ok(build),
        BuildCompat::V5(build) => {
            eprintln!(
                "⚠️  build.toml uses edition 5; upgrading to edition {CURRENT_EDITION} in memory. \
                 Run `kernel-builder update-build` to persist the upgrade."
            );
            Ok(Build::from(build).into())
        }
        BuildCompat::V4(build) => {
            eprintln!(
                "⚠️  build.toml uses the legacy v4 format; upgrading to edition {CURRENT_EDITION} in memory. \
                 Run `kernel-builder update-build` to persist the upgrade."
            );
            Ok(Build::from(build).into())
        }
        BuildCompat::V3(_) => bail!(
            "build.toml uses an unsupported legacy format; migrate it with \
             `kernel-builder update-build`"
        ),
    }
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
