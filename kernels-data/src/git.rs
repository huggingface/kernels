use std::fmt::Display;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use thiserror::Error;

/// A git object identifier.
///
/// The identifier is stored in its canonical (lowercase hexadecimal) form and
/// is validated on construction: it must be a full SHA-1 (40 hexadecimal
/// digits) or SHA-256 (64 hexadecimal digits) identifier. Abbreviated
/// identifiers are rejected.
///
/// The representation is intentionally opaque, use [`Oid::as_str`] or
/// [`Display`] to get at the identifier.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Oid(String);

impl Oid {
    /// The identifier as a lowercase hexadecimal string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Error parsing a git object identifier.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("Invalid git object id, expected 40 or 64 hexadecimal digits: {0}")]
pub struct OidError(String);

impl FromStr for Oid {
    type Err = OidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let valid_length = matches!(s.len(), 40 | 64);
        if !valid_length || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(OidError(s.to_owned()));
        }
        Ok(Oid(s.to_ascii_lowercase()))
    }
}

impl AsRef<str> for Oid {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Display for Oid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'de> Deserialize<'de> for Oid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Oid::from_str(&s).map_err(de::Error::custom)
    }
}

impl Serialize for Oid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

/// The state of a git working tree.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct GitStatus {
    /// Identifier of the `HEAD` commit.
    ///
    /// `sha` is accepted as an alias when deserializing, since older metadata
    /// used that name.
    #[serde(alias = "sha")]
    pub commit: Oid,

    /// Whether the working tree had uncommitted changes to tracked files.
    pub dirty: bool,
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{GitStatus, Oid};

    const SHA1: &str = "d0610aa58db33b142c86b59598a2a1c730f52996";
    const SHA256: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

    #[test]
    fn sha1_and_sha256_oids_are_accepted() {
        assert_eq!(Oid::from_str(SHA1).unwrap().as_str(), SHA1);
        assert_eq!(Oid::from_str(SHA256).unwrap().as_str(), SHA256);
    }

    #[test]
    fn oids_are_normalized_to_lowercase() {
        let oid = Oid::from_str(&SHA1.to_ascii_uppercase()).unwrap();
        assert_eq!(oid.as_str(), SHA1);
        assert_eq!(oid, Oid::from_str(SHA1).unwrap());
    }

    #[test]
    fn invalid_oids_are_rejected() {
        for invalid in [
            "",
            "d0610aa",
            &SHA1[..39],
            &format!("{SHA1}0"),
            &format!("{SHA256}0"),
            &SHA1.replace('d', "z"),
            &format!("{} ", &SHA1[..39]),
        ] {
            assert!(
                Oid::from_str(invalid).is_err(),
                "should be rejected: {invalid:?}"
            );
        }
    }

    #[test]
    fn oid_displays_as_hex() {
        assert_eq!(Oid::from_str(SHA1).unwrap().to_string(), SHA1);
    }

    #[test]
    fn oid_round_trips_through_json() {
        let oid = Oid::from_str(SHA1).unwrap();
        let json = serde_json::to_string(&oid).unwrap();
        assert_eq!(json, format!("\"{SHA1}\""));
        assert_eq!(serde_json::from_str::<Oid>(&json).unwrap(), oid);
    }

    #[test]
    fn invalid_oid_fails_to_deserialize() {
        assert!(serde_json::from_str::<Oid>("\"not-an-oid\"").is_err());
        assert!(serde_json::from_str::<Oid>("42").is_err());
    }

    #[test]
    fn git_status_round_trips_through_json() {
        let status = GitStatus {
            commit: Oid::from_str(SHA1).unwrap(),
            dirty: true,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, format!(r#"{{"commit":"{SHA1}","dirty":true}}"#));
        assert_eq!(serde_json::from_str::<GitStatus>(&json).unwrap(), status);
    }

    #[test]
    fn git_status_accepts_legacy_sha_field() {
        let json = format!(r#"{{"sha":"{SHA1}","dirty":false}}"#);
        let status: GitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status.commit, Oid::from_str(SHA1).unwrap());
        assert!(!status.dirty);
    }
}
