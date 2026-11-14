use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use eyre::{Context, Result};
use kernels_common::git::{GitStatus, Oid};
use kernels_common::metadata::{Metadata, Provenance};

use crate::pyproject::common;
use crate::util::{check_or_infer_kernel_dir, discover_variants};

/// Set the provenance in the metadata of the build variants of a kernel or
/// SDK.
///
/// The kernel provenance needs to be passed explicitly, since we cannot
/// infer it from paths that are not a Git repo (e.g. Nix builds). Any
/// existing provenance information is overwritten.
pub fn add_provenance(
    kernel_dir: Option<PathBuf>,
    kernel_sha: Option<Oid>,
    kernel_dirty: bool,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let (_, variants) = discover_variants(&kernel_dir)?;

    let kernel = kernel_sha.map(|commit| GitStatus {
        commit,
        dirty: kernel_dirty,
    });
    let provenance = common::provenance(kernel);

    for variant in variants {
        eprintln!(
            "Writing provenance to variant `{}`...",
            variant.file_name().unwrap().to_string_lossy()
        );
        set_provenance(&variant, &provenance)?;
    }

    Ok(())
}

fn set_provenance(variant: &Path, provenance: &Provenance) -> Result<()> {
    let metadata_path = variant.join("metadata.json");

    let f = File::open(&metadata_path).context(format!(
        "Cannot open `{}` for reading",
        metadata_path.to_string_lossy()
    ))?;
    let mut metadata: Metadata = serde_json::from_reader(BufReader::new(f)).context(format!(
        "Cannot parse `{}`",
        metadata_path.to_string_lossy()
    ))?;

    metadata.provenance = Some(provenance.clone());

    let f = File::create(&metadata_path).context(format!(
        "Cannot open `{}` for writing provenance",
        metadata_path.to_string_lossy()
    ))?;
    serde_json::to_writer_pretty(BufWriter::new(f), &metadata).context(format!(
        "Cannot write updated metadata to `{}`",
        metadata_path.to_string_lossy()
    ))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::str::FromStr;

    use super::*;

    const KERNEL_SHA: &str = "d0610aa58db33b142c86b59598a2a1c730f52996";

    const METADATA: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 1, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}}"#;

    // Provenance that is not written by the builder.
    const METADATA_WITH_PROVENANCE: &str = r#"{"name": "test-kernel", "id": "kernel_id", "version": 1, "license": "Apache-2.0", "python-depends": [], "backend": {"type": "cuda"}, "provenance": {"kernel-builder": {"version": "0.0.0", "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "dirty": false}, "kernel": {"commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "dirty": false}}}"#;

    fn write_variant(dir: &Path, name: &str, metadata: &str) -> PathBuf {
        let variant = dir.join(name);
        fs::create_dir_all(&variant).unwrap();
        fs::write(variant.join("metadata.json"), metadata).unwrap();
        variant
    }

    fn read_metadata(variant: &Path) -> Metadata {
        let f = File::open(variant.join("metadata.json")).unwrap();
        serde_json::from_reader(BufReader::new(f)).unwrap()
    }

    #[test]
    fn adds_kernel_and_builder_provenance() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cuda = write_variant(temp_dir.path(), "torch-cuda", METADATA);
        let cpu = write_variant(temp_dir.path(), "torch-cpu", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            Some(Oid::from_str(KERNEL_SHA).unwrap()),
            true,
        )
        .unwrap();

        for variant in [cuda, cpu] {
            let provenance = read_metadata(&variant).provenance.unwrap();

            let kernel = provenance.kernel.unwrap();
            assert_eq!(kernel.commit.as_str(), KERNEL_SHA);
            assert!(kernel.dirty);

            // The builder part comes from this binary.
            assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
        }
    }

    #[test]
    fn clean_kernel_provenance() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(
            Some(temp_dir.path().to_path_buf()),
            Some(Oid::from_str(KERNEL_SHA).unwrap()),
            false,
        )
        .unwrap();

        let kernel = read_metadata(&variant).provenance.unwrap().kernel.unwrap();
        assert!(!kernel.dirty);
    }

    #[test]
    fn kernel_provenance_is_omitted_without_sha() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(Some(temp_dir.path().to_path_buf()), None, false).unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(provenance.kernel.is_none());
        assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn existing_provenance_is_replaced() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA_WITH_PROVENANCE);

        // Without a kernel commit, the kernel provenance of the metadata must
        // not survive either.
        add_provenance(Some(temp_dir.path().to_path_buf()), None, false).unwrap();

        let provenance = read_metadata(&variant).provenance.unwrap();
        assert!(provenance.kernel.is_none());
        assert_eq!(provenance.kernel_builder.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn other_metadata_is_preserved() {
        let temp_dir = tempfile::tempdir().unwrap();
        let variant = write_variant(temp_dir.path(), "torch-cuda", METADATA);

        add_provenance(Some(temp_dir.path().to_path_buf()), None, false).unwrap();

        let metadata = read_metadata(&variant);
        assert_eq!(metadata.id, "kernel_id");
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.license, "Apache-2.0");
    }
}
