use std::{
    env::current_dir,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use eyre::{ensure, Context, Result};

use kernels_data::config::{Build, BuildCompat};

pub(crate) fn parse_build(kernel_dir: impl AsRef<Path>) -> Result<Build> {
    let build_compat = parse_and_validate(kernel_dir)?;

    if matches!(build_compat, BuildCompat::V1(_) | BuildCompat::V2(_)) {
        eprintln!(
            "build.toml is in the deprecated V1 or V2 format, use `kernel-builder update-build` to update."
        )
    }

    let build: Build = build_compat
        .try_into()
        .context("Cannot update build configuration")?;

    Ok(build)
}

pub(crate) fn check_or_infer_kernel_dir(kernel_dir: Option<PathBuf>) -> Result<PathBuf> {
    match kernel_dir {
        Some(kernel_dir) => {
            ensure!(
                kernel_dir.is_dir(),
                "`{}` is not a directory",
                kernel_dir.to_string_lossy()
            );
            Ok(kernel_dir)
        }
        None => Ok(current_dir()?),
    }
}

pub(crate) fn check_or_infer_target_dir(
    kernel_dir: impl AsRef<Path>,
    target_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    let kernel_dir = kernel_dir.as_ref();
    match target_dir {
        Some(target_dir) => {
            ensure!(
                target_dir.is_dir(),
                "`{}` is not a directory",
                target_dir.to_string_lossy()
            );
            Ok(target_dir)
        }
        None => Ok(std::path::absolute(kernel_dir)?),
    }
}

pub(crate) fn parse_and_validate(kernel_dir: impl AsRef<Path>) -> Result<BuildCompat> {
    let build_toml = kernel_dir.as_ref().join("build.toml");
    let mut toml_data = String::new();
    File::open(&build_toml)
        .wrap_err_with(|| format!("Cannot open {} for reading", build_toml.to_string_lossy()))?
        .read_to_string(&mut toml_data)
        .wrap_err_with(|| format!("Cannot read from {}", build_toml.to_string_lossy()))?;

    let build_compat: BuildCompat = toml::from_str(&toml_data)
        .wrap_err_with(|| format!("Cannot parse TOML in {}", build_toml.to_string_lossy()))?;

    Ok(build_compat)
}
