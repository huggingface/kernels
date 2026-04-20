use std::str::FromStr;

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::config::{Backend, KernelName};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archs: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
/// Struct for metadata serialization.
///
/// This strict is more strict than `Metadata` and contains what we
/// always want to be written in the current kernels version.
pub struct MetadataStrict {
    pub name: KernelName,
    pub id: String,
    pub version: usize,
    pub license: String,
    pub upstream: Option<url::Url>,
    pub python_depends: Vec<String>,
    pub backend: BackendInfo,
}

/// Kernel metadata.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    pub name: KernelName,
    pub id: Option<String>,
    pub version: Option<usize>,
    pub license: Option<String>,
    pub upstream: Option<url::Url>,
    pub python_depends: Vec<String>,
    pub backend: BackendInfo,
}

impl Metadata {
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
