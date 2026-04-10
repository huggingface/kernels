use std::{collections::BTreeMap, fs, path::Path};

use base64::prelude::{BASE64_STANDARD, Engine as _};
use digest::{Digest, DynDigest};
use eyre::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Sha512};
use walkdir::WalkDir;

use crate::config::Backend;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<url::Url>,
    pub python_depends: Vec<String>,
    pub backend: BackendInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_digest: Option<SourceDigest>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SourceDigest {
    algorithm: DigestAlgorithm,
    files: BTreeMap<String, String>,
}

impl SourceDigest {
    pub fn update_hashes(
        digest_algorithm: DigestAlgorithm,
        variant_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let variant_path = variant_path.as_ref();

        let mut files = BTreeMap::new();
        for entry in WalkDir::new(variant_path) {
            let entry = entry.wrap_err("Failed to read directory entry for hashing")?;
            if !entry.file_type().is_file()
                || entry.path().extension().and_then(|e| e.to_str()) == Some("pyc")
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

        Ok(SourceDigest {
            files,
            algorithm: digest_algorithm,
        })
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub enum DigestAlgorithm {
    #[serde(rename = "sha256")]
    SHA256,

    #[serde(rename = "sha512")]
    SHA512,
}

impl From<DigestAlgorithm> for Box<dyn DynDigest> {
    fn from(digest_algorithm: DigestAlgorithm) -> Box<dyn DynDigest> {
        match digest_algorithm {
            DigestAlgorithm::SHA256 => Box::new(Sha256::new()),
            DigestAlgorithm::SHA512 => Box::new(Sha512::new()),
        }
    }
}

pub fn parse_metadata(path: impl AsRef<Path>) -> Result<Metadata> {
    let path = path.as_ref();
    let data =
        fs::read_to_string(path).wrap_err_with(|| format!("Cannot read `{}`", path.display()))?;
    serde_json::from_str(&data).wrap_err_with(|| format!("Cannot parse `{}`", path.display()))
}
