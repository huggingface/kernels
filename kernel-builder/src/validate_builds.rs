use std::fs;
use std::path::PathBuf;
use std::{fs::File, io::BufReader};

use eyre::{Context, Result};
use kernels_data::metadata::Metadata;

use crate::util::{check_or_infer_kernel_dir, discover_variants};

pub fn check_builds(kernel_dir: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let kernel_dir = fs::canonicalize(&kernel_dir)
        .wrap_err_with(|| format!("Cannot resolve kernel directory `{}`", kernel_dir.display()))?;

    let (build_dir, variants) = discover_variants(&kernel_dir)?;
    for variant in variants {
        let variant_path = build_dir.join(&variant);
        check_metadata(&variant_path)?;
    }

    Ok(())
}

fn check_metadata(variant_path: &PathBuf) -> Result<()> {
    let metadata_path = variant_path.join("metadata.json");
    if !metadata_path.exists() {
        eyre::bail!("Metadata file not found for variant at {:?}", variant_path);
    }

    let f = File::open(&metadata_path)?;
    let _metadata: Metadata = serde_json::from_reader(BufReader::new(f)).context(format!(
        "Failed to parse metadata: {}",
        metadata_path.to_string_lossy()
    ))?;

    Ok(())
}
