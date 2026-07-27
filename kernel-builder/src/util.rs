use std::env::current_dir;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use eyre::{bail, ensure, Context, Result};

use kernels_data::config::{Build, BuildCompat, CurrentConfig, Kernel};

pub(crate) fn parse_build(kernel_dir: impl AsRef<Path>) -> Result<Build> {
    let kernel_dir = kernel_dir.as_ref();
    let mut build: Build = parse_and_validate(kernel_dir)?.into();
    infer_rust_kernel_lib_names(&mut build, kernel_dir)?;
    Ok(build)
}

pub(crate) fn check_or_infer_kernel_dir(kernel_dir: Option<impl AsRef<Path>>) -> Result<PathBuf> {
    match kernel_dir {
        Some(kernel_dir) => {
            let kernel_dir = kernel_dir.as_ref();
            ensure!(
                kernel_dir.is_dir(),
                "`{}` is not a directory",
                kernel_dir.to_string_lossy()
            );
            Ok(kernel_dir.to_owned())
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

pub(crate) fn parse_and_validate(kernel_dir: impl AsRef<Path>) -> Result<CurrentConfig> {
    // Only v4 is auto-upgraded to v5 on load; older editions must be migrated
    // explicitly with `update-build`.
    match parse_and_validate_compat(kernel_dir)? {
        BuildCompat::V5(build) => Ok(build),
        BuildCompat::V4(build) => {
            eprintln!(
                "⚠️  build.toml uses the legacy v4 format; upgrading to edition 5 in memory. \
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

fn infer_rust_kernel_lib_names(build: &mut Build, kernel_dir: &Path) -> Result<()> {
    for (kernel_name, kernel) in &mut build.kernels {
        let (src, lib_name) = match kernel {
            Kernel::RustCpu { src, lib_name, .. } | Kernel::RustCuda { src, lib_name, .. } => {
                (src, lib_name)
            }
            _ => continue,
        };
        if lib_name.is_none() {
            let manifest_path = src
                .iter()
                .find(|path| {
                    Path::new(path.as_str())
                        .file_name()
                        .is_some_and(|name| name == "Cargo.toml")
                })
                .map(|path| kernel_dir.join(path))
                .ok_or_else(|| {
                    eyre::eyre!(
                        "Rust kernel `{kernel_name}`: `src` must include the crate Cargo.toml"
                    )
                })?;
            let manifest: toml::Value = toml::from_str(&fs::read_to_string(&manifest_path)?)?;
            let name = |table| manifest.get(table)?.get("name")?.as_str();
            *lib_name = Some(
                name("lib")
                    .or_else(|| name("package"))
                    .ok_or_else(|| {
                        eyre::eyre!("cannot infer `lib-name` from `{}`", manifest_path.display())
                    })?
                    .replace('-', "_"),
            );
        }
    }
    Ok(())
}

/// Discover build variant directories (contain `metadata.json`).
/// Checks `result` symlink (Nix store output) first, then falls back to `build/`.
pub(crate) fn discover_variants(kernel_dir: &Path) -> Result<(PathBuf, Vec<PathBuf>)> {
    let candidates = [
        kernel_dir.join("result"),
        kernel_dir.join("build"),
        kernel_dir.to_path_buf(),
    ];

    for candidate in &candidates {
        if !candidate.is_dir() {
            continue;
        }

        let mut variants: Vec<PathBuf> = fs::read_dir(candidate)
            .wrap_err_with(|| format!("Cannot read `{}`", candidate.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.join("metadata.json").is_file())
            .collect();

        if !variants.is_empty() {
            variants.sort();
            return Ok((candidate.clone(), variants));
        }
    }

    bail!(
        "No build variants found in `{}`, `{}`, or `{}`",
        candidates[0].display(),
        candidates[1].display(),
        candidates[2].display(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_variants() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        let build_dir = kernel_dir.join("build");
        fs::create_dir_all(build_dir.join("variant-a")).unwrap();
        fs::create_dir_all(build_dir.join("variant-b")).unwrap();

        fs::write(
            build_dir.join("variant-a/metadata.json"),
            r#"{"version": 1}"#,
        )
        .unwrap();
        fs::write(
            build_dir.join("variant-b/metadata.json"),
            r#"{"version": 1}"#,
        )
        .unwrap();

        let (found_build_dir, variants) = discover_variants(kernel_dir).unwrap();
        assert_eq!(found_build_dir, build_dir);
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn test_discover_variants_no_variants() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = discover_variants(temp_dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_discover_variants_from_result_symlink() {
        let temp_dir = tempfile::tempdir().unwrap();
        let kernel_dir = temp_dir.path();

        // Create a "nix store" directory with variants
        let store_dir = kernel_dir.join("nix-store-output");
        fs::create_dir_all(store_dir.join("variant-a")).unwrap();
        fs::write(
            store_dir.join("variant-a/metadata.json"),
            r#"{"version": 1}"#,
        )
        .unwrap();

        // Create result symlink pointing to store output
        #[cfg(unix)]
        std::os::unix::fs::symlink(&store_dir, kernel_dir.join("result")).unwrap();

        #[cfg(unix)]
        {
            let (found_dir, variants) = discover_variants(kernel_dir).unwrap();
            assert_eq!(found_dir, kernel_dir.join("result"));
            assert_eq!(variants.len(), 1);
        }
    }
}
