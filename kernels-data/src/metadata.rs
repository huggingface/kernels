use std::str::FromStr;

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::config::{Backend, GitUrl, KernelName};
use crate::digest::Digest;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archs: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct GitInfo {
    pub sha: String,
    pub dirty: bool,
}

/// Provenance of the `kernel-builder` that produced a build.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct KernelBuilderInfo {
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
    pub dirty: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BuildInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_builder: Option<KernelBuilderInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<GitInfo>,
}

impl BuildInfo {
    /// Whether either the `kernel-builder` or the kernel source was dirty.
    pub fn is_dirty(&self) -> bool {
        self.kernel_builder.as_ref().is_some_and(|kb| kb.dirty)
            || self.kernel.as_ref().is_some_and(|k| k.dirty)
    }
}

/// Kernel metadata.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    pub name: KernelName,
    pub id: String,
    pub version: usize,
    pub license: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<GitUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<GitUrl>,
    pub python_depends: Vec<String>,
    pub backend: BackendInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<Digest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_info: Option<BuildInfo>,
}

impl Metadata {
    /// Read the metadata from a JSON byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Read the metadata from a JSON `std::io::Read`.
    pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }
}

impl FromStr for Metadata {
    type Err = eyre::Report;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METADATA_NO_BUILD_INFO: &str = r#"{
        "name": "relu",
        "id": "_relu_cuda_abc1234",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": { "type": "cuda" }
    }"#;

    const METADATA_WITH_BUILD_INFO: &str = r#"{
        "name": "relu",
        "id": "_relu_cuda_abc1234",
        "version": 1,
        "license": "Apache-2.0",
        "python-depends": [],
        "backend": { "type": "cuda" },
        "build-info": {
            "kernel-builder": { "version": "0.16.0-dev0", "sha": "1111111111111111111111111111111111111111", "dirty": true },
            "kernel": { "sha": "2222222222222222222222222222222222222222", "dirty": false }
        }
    }"#;

    #[test]
    fn parses_metadata_without_build_info() {
        let metadata: Metadata = METADATA_NO_BUILD_INFO.parse().unwrap();
        assert!(metadata.build_info.is_none());
    }

    #[test]
    fn parses_and_reports_dirty_build_info() {
        let metadata: Metadata = METADATA_WITH_BUILD_INFO.parse().unwrap();
        let build_info = metadata.build_info.expect("build-info should be present");
        assert!(build_info.is_dirty());

        let kernel_builder = build_info.kernel_builder.unwrap();
        assert_eq!(kernel_builder.version, "0.16.0-dev0");
        assert!(kernel_builder.dirty);
        assert_eq!(kernel_builder.sha.as_deref(), Some(&"1".repeat(40)[..]));

        let kernel = build_info.kernel.unwrap();
        assert!(!kernel.dirty);
        assert_eq!(kernel.sha, "2".repeat(40));
    }

    #[test]
    fn build_info_round_trips_with_kebab_case_keys() {
        let metadata: Metadata = METADATA_WITH_BUILD_INFO.parse().unwrap();
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("\"build-info\""));
        assert!(json.contains("\"kernel-builder\""));

        // Re-parsing the serialized form yields the same dirtiness.
        let reparsed: Metadata = json.parse().unwrap();
        assert!(reparsed.build_info.unwrap().is_dirty());
    }

    #[test]
    fn build_info_is_not_dirty_when_all_clean() {
        let build_info = BuildInfo {
            kernel_builder: Some(KernelBuilderInfo {
                version: "0.16.0".to_owned(),
                sha: None,
                dirty: false,
            }),
            kernel: Some(GitInfo {
                sha: "abc".to_owned(),
                dirty: false,
            }),
        };
        assert!(!build_info.is_dirty());
    }
}
