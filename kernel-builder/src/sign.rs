use std::{fs::File, io::BufReader, path::PathBuf};

use eyre::Result;
use kernels_data::metadata::{DigestAlgorithm, Metadata, SourceDigest};

use crate::util::{check_or_infer_kernel_dir, discover_variants};

pub fn sign(kernel_dir: Option<PathBuf>) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let (build_dir, variants) = discover_variants(&kernel_dir)?;

    for variant in variants {
        eprintln!(
            "Signing variant `{}`...",
            variant.file_name().unwrap().to_string_lossy()
        );

        let f = File::open(build_dir.join(&variant).join("metadata.json"))?;
        let metadata: Metadata = serde_json::from_reader(BufReader::new(f))?;
        let source_digest =
            SourceDigest::update_hashes(DigestAlgorithm::SHA256, build_dir.join(variant))?;

        eprintln!("{}", serde_json::to_string_pretty(&source_digest)?);
    }

    Ok(())
}
