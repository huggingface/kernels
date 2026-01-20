use std::{fmt::Display, str::FromStr};

use eyre::{ensure, Context};
use itertools::Itertools;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Version(Vec<usize>);

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(de::Error::custom)
    }
}

impl Serialize for Version {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0.iter().map(|v| v.to_string()).join("."))
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            itertools::join(self.0.iter().map(|v| v.to_string()), ".")
        )
    }
}

impl From<Vec<usize>> for Version {
    fn from(value: Vec<usize>) -> Self {
        // Remove trailing zeros for normalization.
        let mut normalized = value
            .into_iter()
            .rev()
            .skip_while(|&x| x == 0)
            .collect::<Vec<_>>();
        normalized.reverse();
        Version(normalized)
    }
}

impl FromStr for Version {
    type Err = eyre::Report;

    fn from_str(version: &str) -> Result<Self, Self::Err> {
        let version = version.trim().to_owned();
        ensure!(!version.is_empty(), "Empty version string");
        let mut version_parts = Vec::new();
        for part in version.split('.') {
            let version_part: usize = part
                .parse()
                .context(format!("Version must consist of numbers: {version}"))?;
            version_parts.push(version_part);
        }

        Ok(Version::from(version_parts))
    }
}
