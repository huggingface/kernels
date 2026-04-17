use std::{fs, path::Path};

use eyre::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::Backend;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
    pub archs: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<url::Url>,
    pub python_depends: Vec<String>,
    pub backend: BackendInfo,
}

pub fn parse_metadata(path: impl AsRef<Path>) -> Result<Metadata> {
    let path = path.as_ref();
    let data =
        fs::read_to_string(path).wrap_err_with(|| format!("Cannot read `{}`", path.display()))?;
    serde_json::from_str(&data).wrap_err_with(|| format!("Cannot parse `{}`", path.display()))
}
