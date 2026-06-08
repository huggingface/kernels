use std::{collections::BTreeMap, fs, path::Path};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use digest::{Digest as _, DynDigest};
use eyre::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Sha512};
use walkdir::WalkDir;

#[derive(Debug, Deserialize, Serialize)]
pub struct Digest {
    algorithm: DigestAlgorithm,
    files: BTreeMap<String, String>,
}

impl Digest {
    pub fn hash_variant(
        digest_algorithm: DigestAlgorithm,
        variant_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let variant_path = variant_path.as_ref();

        let mut files = BTreeMap::new();
        for entry in WalkDir::new(variant_path) {
            let entry = entry.wrap_err("Failed to read directory entry for hashing")?;
            // Hub cache path can be symlinks.
            let path = fs::canonicalize(entry.path())?;
            if !path.is_file()
                // Metadata file hash cannot be computed, since it stores
                // hashes. It is protected by the detached signature.
                || entry.path().file_name().and_then(|f| f.to_str()) == Some("metadata.json")
                || entry.path().file_name().and_then(|f| f.to_str()) == Some("metadata.json.sigstore")
                // Python likes to create .pyc files in __pycache__/ directories.
                || entry.path().components().any(|c| c.as_os_str() == "__pycache__")
            {
                continue;
            }

            let path = entry.path();

            // Read and hash contents.
            let contents = fs::read(path)
                .wrap_err_with(|| format!("Cannot read `{}` for hashing", path.display()))?;
            let mut hasher: Box<dyn DynDigest> = digest_algorithm.into();
            hasher.update(&contents);

            let relative_path = path.strip_prefix(variant_path).wrap_err_with(|| {
                format!("Cannot strip prefix from `{}` for hashing", path.display())
            })?;

            // Normalize Windows directory separators.
            let relative_path_str = relative_path.to_string_lossy().replace('\\', "/");

            let hash_base64 = BASE64_STANDARD.encode(hasher.finalize_reset());

            files.insert(relative_path_str, hash_base64);
        }

        Ok(Digest {
            files,
            algorithm: digest_algorithm,
        })
    }

    /// Algorithm used for hashing.
    pub fn algorithm(&self) -> DigestAlgorithm {
        self.algorithm
    }

    /// Mapping of relative path -> base64 digest.
    pub fn files(&self) -> &BTreeMap<String, String> {
        &self.files
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum DigestAlgorithm {
    #[serde(rename = "sha256")]
    SHA256,

    #[serde(rename = "sha512")]
    SHA512,
}

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn hash_variant_hashes_file_with_sha256() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.path().join("_extension.so"), b"hello world")?;

        let digest = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;

        assert_eq!(digest.files().len(), 1);
        assert_eq!(
            digest.files().get("_extension.so").map(String::as_str),
            Some("uU0nuZNNPgilLlLX2n2r+sSE7+N6U4DukIj3rOLvzek=")
        );
        Ok(())
    }

    #[test]
    fn hash_variant_hashes_file_with_sha512() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.path().join("_extension.so"), b"hello world")?;

        let digest = Digest::hash_variant(DigestAlgorithm::SHA512, dir.path())?;

        assert_eq!(digest.files().len(), 1);
        assert_eq!(
            digest.files().get("_extension.so").map(String::as_str),
            Some(
                "MJ7MSJwS1utMxA9QyQLytNDtd+5RGnx6m808qG1M2G+YndNbxf9JlnDaNCVbRbDP2DDoH2Bdz33FVC6TrpzXbw=="
            )
        );
        Ok(())
    }

    #[test]
    fn hash_variant_skips_excluded_files() -> Result<()> {
        let dir = TempDir::new()?;
        let pycache_dir = dir.path().join("pkg").join("__pycache__");
        fs::create_dir_all(&pycache_dir)?;
        fs::write(dir.path().join("_extension.so"), b"content")?;
        fs::write(dir.path().join("metadata.json"), b"{}")?;
        fs::write(dir.path().join("metadata.json.sigstore"), b"sig")?;
        fs::write(pycache_dir.join("mod.cpython-311.pyc"), b"bytecode")?;
        // A .pyc outside __pycache__/ must be included — it was not generated
        // locally by Python and should be covered by the digest.
        fs::write(dir.path().join("payload.pyc"), b"smuggled")?;

        let digest = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;

        assert_eq!(digest.files().len(), 2);
        assert!(digest.files().contains_key("_extension.so"));
        assert!(digest.files().contains_key("payload.pyc"));
        assert!(!digest.files().contains_key("metadata.json"));
        assert!(!digest.files().contains_key("metadata.json.sigstore"));
        assert!(
            !digest
                .files()
                .contains_key("pkg/__pycache__/mod.cpython-311.pyc")
        );
        Ok(())
    }

    #[test]
    fn hash_variant_uses_forward_slash_paths() -> Result<()> {
        let dir = TempDir::new()?;
        let subdir = dir.path().join("some").join("subdir");
        fs::create_dir_all(&subdir)?;
        fs::write(subdir.join("__init__.py"), b"data")?;

        let digest = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;

        assert!(digest.files().contains_key("some/subdir/__init__.py"));
        Ok(())
    }

    #[test]
    fn hash_variant_returns_empty_map_for_empty_dir() -> Result<()> {
        let dir = TempDir::new()?;

        let digest = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;

        assert!(digest.files().is_empty());
        Ok(())
    }

    #[test]
    fn hash_variant_produces_deterministic_results() -> Result<()> {
        let dir = TempDir::new()?;
        fs::write(dir.path().join("_extension.so"), b"hello world")?;

        let digest1 = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;
        let digest2 = Digest::hash_variant(DigestAlgorithm::SHA256, dir.path())?;

        assert_eq!(digest1.files(), digest2.files());
        Ok(())
    }
}

impl From<DigestAlgorithm> for Box<dyn DynDigest> {
    fn from(digest_algorithm: DigestAlgorithm) -> Box<dyn DynDigest> {
        match digest_algorithm {
            DigestAlgorithm::SHA256 => Box::new(Sha256::new()),
            DigestAlgorithm::SHA512 => Box::new(Sha512::new()),
        }
    }
}
