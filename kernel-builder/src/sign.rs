use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use eyre::{Context, Result};
use kernels_data::metadata::{DigestAlgorithm, Metadata, SourceDigest};

use crate::util::{check_or_infer_kernel_dir, discover_variants};

pub fn sign(kernel_dir: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let (build_dir, variants) = discover_variants(&kernel_dir)?;

    for variant in variants {
        let variant_path = build_dir.join(&variant);
        let metadata_path = variant_path.join("metadata.json");

        eprintln!(
            "Signing variant `{}`...",
            variant.file_name().unwrap().to_string_lossy()
        );

        let f = File::open(&metadata_path).context(format!(
            "Cannot open `{}` for reading",
            metadata_path.to_string_lossy()
        ))?;
        let mut metadata: Metadata = serde_json::from_reader(BufReader::new(f))?;

        let source_digest = SourceDigest::update_hashes(DigestAlgorithm::SHA256, &variant_path)?;
        metadata.source_digest = Some(source_digest);

        let f = File::create(&metadata_path).context(format!(
            "Cannot open `{}` for writing file hashes",
            metadata_path.to_string_lossy()
        ))?;
        serde_json::to_writer_pretty(BufWriter::new(f), &metadata).context(format!(
            "Cannot write updated metadata to `{}`",
            metadata_path.to_string_lossy()
        ))?;
    }

    Ok(())
}
