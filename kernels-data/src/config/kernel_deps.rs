use serde::{Deserialize, Serialize, de};

/// Kernel version (numeric or Git revision).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged, rename_all = "kebab-case")]
pub enum KernelVersion {
    Version(usize),
    Revision(String),
}

/// A kernel dependency.
#[derive(Clone, Debug)]
pub struct KernelDependency {
    pub repo_id: String,
    pub version: KernelVersion,
}

// Note: enum flattening + denying unknown fields is recommended against:
//
// https://serde.rs/field-attrs.html#flatten
//
// So we roll a bit of extra serde code to support both
//
// { repo-id = "foo/bar", version = 1 }
// { repo-id = "foo/bar", revision = "somerevision" }
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct KernelDependencyRepr {
    repo_id: String,
    version: Option<usize>,
    revision: Option<String>,
}

impl<'de> Deserialize<'de> for KernelDependency {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let repr = KernelDependencyRepr::deserialize(deserializer)?;
        let version = match (repr.version, repr.revision) {
            (Some(v), None) => KernelVersion::Version(v),
            (None, Some(r)) => KernelVersion::Revision(r),
            (Some(_), Some(_)) => {
                return Err(de::Error::custom(
                    "`version` and `revision` are mutually exclusive",
                ));
            }
            (None, None) => {
                return Err(de::Error::custom(
                    "either `version` or `revision` must be specified",
                ));
            }
        };
        Ok(KernelDependency {
            repo_id: repr.repo_id,
            version,
        })
    }
}

impl Serialize for KernelDependency {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let repr = match &self.version {
            KernelVersion::Version(v) => KernelDependencyRepr {
                repo_id: self.repo_id.clone(),
                version: Some(*v),
                revision: None,
            },
            KernelVersion::Revision(r) => KernelDependencyRepr {
                repo_id: self.repo_id.clone(),
                version: None,
                revision: Some(r.clone()),
            },
        };
        repr.serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_version() {
        let dep = KernelDependency {
            repo_id: "kernels-staging/einops".to_string(),
            version: KernelVersion::Version(1),
        };
        let serialized = toml::to_string(&dep).unwrap();
        assert!(!serialized.contains("revision"));

        let parsed: KernelDependency = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed.repo_id, dep.repo_id);
        assert!(matches!(parsed.version, KernelVersion::Version(1)));
    }

    #[test]
    fn roundtrip_revision() {
        let dep = KernelDependency {
            repo_id: "kernels-staging/einops".to_string(),
            version: KernelVersion::Revision("34fa".to_string()),
        };
        let serialized = toml::to_string(&dep).unwrap();
        assert!(!serialized.contains("version"));

        let parsed: KernelDependency = toml::from_str(&serialized).unwrap();
        assert_eq!(parsed.repo_id, dep.repo_id);
        assert!(matches!(
            parsed.version,
            KernelVersion::Revision(ref r) if r == "34fa"
        ));
    }

    #[test]
    fn version_string_is_rejected() {
        let toml_str = r#"
repo-id = "x"
version = "notanint"
"#;
        assert!(toml::from_str::<KernelDependency>(toml_str).is_err());
    }

    #[test]
    fn both_version_and_revision_is_rejected() {
        let toml_str = r#"
repo-id = "x"
version = 1
revision = "abc"
"#;
        assert!(toml::from_str::<KernelDependency>(toml_str).is_err());
    }

    #[test]
    fn neither_version_nor_revision_is_rejected() {
        let toml_str = r#"
repo-id = "x"
"#;
        assert!(toml::from_str::<KernelDependency>(toml_str).is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let toml_str = r#"
repo-id = "x"
version = 1
frobnicate = true
"#;
        assert!(toml::from_str::<KernelDependency>(toml_str).is_err());
    }
}
