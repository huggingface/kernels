use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    pub python_depends: Vec<String>,
}
