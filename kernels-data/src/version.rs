use std::ops::Deref;
use std::{fmt::Display, str::FromStr};

use eyre::{Context, ensure};
use itertools::Itertools;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Version<const N: usize>([usize; N]);

impl<'de, const N: usize> Deserialize<'de> for Version<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        FromStr::from_str(&s).map_err(de::Error::custom)
    }
}

impl<const N: usize> Serialize for Version<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<const N: usize> Display for Version<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().map(|v| v.to_string()).join("."))
    }
}

impl<const N: usize> Deref for Version<N> {
    type Target = [usize];

    #[inline]
    fn deref(&self) -> &[usize] {
        &self.0
    }
}

impl<const N: usize> TryFrom<Vec<usize>> for Version<N> {
    type Error = eyre::Report;

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        ensure!(
            value.len() == N,
            "Version has {} components, expected {N}",
            value.len()
        );
        let mut parts = [0; N];
        parts.copy_from_slice(&value);
        Ok(Version(parts))
    }
}

impl<const N: usize> FromStr for Version<N> {
    type Err = eyre::Report;

    fn from_str(version: &str) -> Result<Self, Self::Err> {
        let version = version.trim();
        ensure!(!version.is_empty(), "Empty version string");
        let mut version_parts = Vec::new();
        for part in version.split('.') {
            let version_part: usize = part
                .parse()
                .context(format!("Version must consist of numbers: {version}"))?;
            version_parts.push(version_part);
        }

        Version::try_from(version_parts)
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::str::FromStr;

    use super::Version;

    #[test]
    fn versions_must_have_exactly_n_components() {
        assert!(Version::<2>::from_str("12.8").is_ok());
        assert!(Version::<2>::from_str("13").is_err());
        assert!(Version::<2>::from_str("12.8.0").is_err());
    }

    #[test]
    fn version_ord() {
        assert_eq!(
            Version::<2>::from_str("5.0")
                .unwrap()
                .cmp(&Version::from_str("5.0").unwrap()),
            Ordering::Equal
        );
        assert_eq!(
            Version::<2>::from_str("5.0")
                .unwrap()
                .cmp(&Version::from_str("5.1").unwrap()),
            Ordering::Less
        );
        assert_eq!(
            Version::<2>::from_str("5.1")
                .unwrap()
                .cmp(&Version::from_str("5.0").unwrap()),
            Ordering::Greater
        );
    }

    #[test]
    fn display_keeps_all_components() {
        assert_eq!(Version::<2>::from_str("12.0").unwrap().to_string(), "12.0");
    }

    #[test]
    fn serde_roundtrip() {
        let version = Version::<2>::from_str("12.8").unwrap();
        let serialized = serde_json::to_string(&version).unwrap();
        assert_eq!(serialized, "\"12.8\"");
        let deserialized: Version<2> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(version, deserialized);
    }
}
