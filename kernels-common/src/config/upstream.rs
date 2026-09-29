//! Compatibility encoding for upstream repositories in build and metadata files.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::GitUrl;

pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<GitUrl>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Upstream {
        Single(GitUrl),
        Multiple(Vec<GitUrl>),
    }

    // Older metadata may explicitly contain null for an absent upstream.
    Ok(match Option::<Upstream>::deserialize(deserializer)? {
        None => vec![],
        Some(Upstream::Single(url)) => vec![url],
        Some(Upstream::Multiple(urls)) => urls,
    })
}

pub fn serialize<S>(upstreams: &[GitUrl], serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // Keep single-source kernels readable by older versions of `kernels`.
    match upstreams {
        [url] => url.serialize(serializer),
        urls => urls.serialize(serializer),
    }
}
