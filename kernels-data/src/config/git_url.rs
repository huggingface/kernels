use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use url::Url;

const GIT_SCHEMES: &[&str] = &[
    "https",
    "http",
    "git",
    "ssh",
    "git+ssh",
    "git+https",
    "git+http",
];

/// A validated git repository URL.
///
/// Accepts standard git URLs (`https://`, `ssh://`, etc.) and SCP-like
/// syntax (`git@github.com:org/repo.git`).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct GitUrl {
    url: Url,
    original: String,
}

impl GitUrl {
    pub fn parse(s: &str) -> Result<Self, String> {
        // Standard URL.
        if let Ok(url) = Url::parse(s) {
            if !GIT_SCHEMES.contains(&url.scheme()) {
                return Err(format!(
                    "unsupported scheme `{}` in git URL `{s}`, expected one of: {}",
                    url.scheme(),
                    GIT_SCHEMES.join(", ")
                ));
            }
            return Ok(Self {
                url,
                original: s.to_string(),
            });
        }

        // SCP-like syntax: git@github.com:org/repo.git
        if let Some((host, path)) = s.split_once(':') {
            if !host.is_empty() && !path.is_empty() && !path.starts_with('/') {
                let normalized = format!("ssh://{host}/{path}");
                if let Ok(url) = Url::parse(&normalized) {
                    return Ok(Self {
                        url,
                        original: s.to_string(),
                    });
                }
            }
        }

        Err(format!("invalid git URL `{s}`"))
    }

    pub fn as_url(&self) -> &Url {
        &self.url
    }
}

impl fmt::Display for GitUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.original)
    }
}

impl<'de> Deserialize<'de> for GitUrl {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        GitUrl::parse(&s).map_err(de::Error::custom)
    }
}

impl Serialize for GitUrl {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.original)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_urls() {
        let valid = [
            "https://github.com/org/repo",
            "https://github.com/org/repo.git",
            "http://github.com/org/repo.git",
            "git://github.com/org/repo.git",
            "ssh://git@github.com/org/repo.git",
            "git+ssh://git@github.com/org/repo.git",
            "git+https://github.com/org/repo.git",
            "git@github.com:org/repo.git",
            "git@github.com:drbh/yamoe.git",
            "user@gitlab.com:group/project.git",
        ];
        for s in valid {
            assert!(GitUrl::parse(s).is_ok(), "expected valid: {s}");
        }
    }

    #[test]
    fn test_scp_normalization() {
        let git_url = GitUrl::parse("git@github.com:drbh/yamoe.git").unwrap();
        assert_eq!(
            git_url.as_url().as_str(),
            "ssh://git@github.com/drbh/yamoe.git"
        );
        assert_eq!(git_url.to_string(), "git@github.com:drbh/yamoe.git");
    }

    #[test]
    fn test_scp_roundtrip() {
        let git_url = GitUrl::parse("git@github.com:org/repo.git").unwrap();
        let json = serde_json::to_string(&git_url).unwrap();
        let deserialized: GitUrl = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.to_string(), "git@github.com:org/repo.git");
    }

    #[test]
    fn test_invalid_urls() {
        assert!(GitUrl::parse("ftp://example.com/repo.git").is_err());
        assert!(GitUrl::parse("file:///home/user/repo").is_err());
        assert!(GitUrl::parse("not a url").is_err());
    }
}
