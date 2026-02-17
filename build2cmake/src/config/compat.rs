use eyre::Result;
use serde::Deserialize;
use serde_value::Value;

use super::{v1, v2, v3, Build};

#[derive(Debug)]
pub enum BuildCompat {
    V1(v1::Build),
    V2(v2::Build),
    V3(v3::Build),
}

impl<'de> Deserialize<'de> for BuildCompat {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        v1::Build::deserialize(value.clone())
            .map(BuildCompat::V1)
            .or_else(|_| v2::Build::deserialize(value.clone()).map(BuildCompat::V2))
            .or_else(|_| v3::Build::deserialize(value.clone()).map(BuildCompat::V3))
            .map_err(serde::de::Error::custom)
    }
}

impl BuildCompat {
    /// Get the kernel name from any build version.
    pub fn name(&self) -> &str {
        match self {
            BuildCompat::V1(build) => &build.general.name,
            BuildCompat::V2(build) => &build.general.name,
            BuildCompat::V3(build) => &build.general.name,
        }
    }
}

impl TryFrom<BuildCompat> for Build {
    type Error = eyre::Error;

    fn try_from(compat: BuildCompat) -> Result<Self> {
        match compat {
            BuildCompat::V1(v1_build) => v1_build.try_into(),
            BuildCompat::V2(v2_build) => v2_build.try_into(),
            BuildCompat::V3(v3_build) => Ok(v3_build.into()),
        }
    }
}
