use std::{fmt::Display, str::FromStr};

use eyre::{ensure, Context, Result};
use serde::{de, Deserialize, Deserializer};

/// Symbol version.
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

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::Version;

    #[test]
    fn test_version_equals() {
        assert_eq!(Version::from(vec![5, 0, 0]), Version::from(vec![5, 0, 0]));
        assert_eq!(Version::from(vec![5, 0, 0]), Version::from(vec![5]));
        assert_eq!(Version::from(vec![5]), Version::from(vec![5, 0, 0]));
    }

    #[test]
    fn version_ord() {
        assert_eq!(
            Version::from(vec![5, 0, 0]).cmp(&Version::from(vec![5, 0, 0])),
            Ordering::Equal
        );
        assert_eq!(
            Version::from(vec![5, 0, 0]).cmp(&Version::from(vec![5])),
            Ordering::Equal
        );
        assert_eq!(
            Version::from(vec![5]).cmp(&Version::from(vec![5, 0, 0])),
            Ordering::Equal
        );
        assert_eq!(
            Version::from(vec![5, 0, 0]).cmp(&Version::from(vec![5, 0, 1])),
            Ordering::Less
        );
        assert_eq!(
            Version::from(vec![5]).cmp(&Version::from(vec![5, 0, 1])),
            Ordering::Less
        );
        assert_eq!(
            Version::from(vec![5, 0, 1]).cmp(&Version::from(vec![5, 0, 0])),
            Ordering::Greater
        );
        assert_eq!(
            Version::from(vec![5, 0, 1]).cmp(&Version::from(vec![5])),
            Ordering::Greater
        );
    }
}
