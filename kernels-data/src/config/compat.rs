use eyre::Result;
use serde::{Deserialize, de};
use serde_value::Value;

use crate::config::ConfigError;

use super::{Build, v3, v4, v5};

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum BuildCompat {
    V3(v3::Build),
    V4(v4::Build),
    V5(v5::Build),
}

#[derive(Deserialize)]
struct EditionProbe {
    #[serde(default)]
    general: Option<GeneralProbe>,
}

#[derive(Deserialize)]
struct GeneralProbe {
    #[serde(default)]
    edition: Option<u32>,
}

impl<'de> Deserialize<'de> for BuildCompat {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;

        let edition = EditionProbe::deserialize(value.clone())
            .map_err(|err| de::Error::custom(format!("invalid `general.edition`: {err}")))?
            .general
            .and_then(|general| general.edition);

        match edition {
            // Configs with editions
            Some(5) => v5::Build::deserialize(value)
                .map(BuildCompat::V5)
                .map_err(de::Error::custom),

            Some(other) => Err(de::Error::custom(format!(
                "unsupported build edition {other}; upgrade kernel-builder"
            ))),

            // Pre-edition configs: try v4 before v3 - v4 is stricter
            None => match v4::Build::deserialize(value.clone()) {
                Ok(build) => Ok(BuildCompat::V4(build)),
                Err(v4_err) => {
                    v3::Build::deserialize(value)
                        .map(BuildCompat::V3)
                        .map_err(|v3_err| {
                            de::Error::custom(format!(
                                "did not match v4 ({v4_err}) or v3 ({v3_err})"
                            ))
                        })
                }
            },
        }
    }
}

impl TryFrom<BuildCompat> for Build {
    type Error = ConfigError;

    fn try_from(compat: BuildCompat) -> Result<Self, Self::Error> {
        match compat {
            BuildCompat::V3(v3_build) => v3_build.try_into(),
            BuildCompat::V4(v4_build) => Ok(v4_build.into()),
            BuildCompat::V5(v5_build) => Ok(v5_build.into()),
        }
    }
}
