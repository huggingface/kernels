use eyre::Result;
use serde::Deserialize;
use serde_value::Value;

use crate::config::ConfigError;

use super::{Build, v3, v4};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum BuildCompat {
    V3(v3::Build),
    V4(v4::Build),
}

impl<'de> Deserialize<'de> for BuildCompat {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        v3::Build::deserialize(value.clone())
            .map(BuildCompat::V3)
            .or_else(|_| v4::Build::deserialize(value.clone()).map(BuildCompat::V4))
            .map_err(serde::de::Error::custom)
    }
}

impl TryFrom<BuildCompat> for Build {
    type Error = ConfigError;

    fn try_from(compat: BuildCompat) -> Result<Self, Self::Error> {
        match compat {
            BuildCompat::V3(v3_build) => v3_build.try_into(),
            BuildCompat::V4(v4_build) => Ok(v4_build.into()),
        }
    }
}
