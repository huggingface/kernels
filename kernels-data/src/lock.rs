use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::KernelDependency;
use crate::git::Oid;

/// A locked kernel revision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct KernelLock {
    /// Locked git SHA.
    pub commit: Oid,
}

/// Multiple kernel locks keyed by the dependency they resolve.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct KernelLocks {
    #[serde(with = "dependency_map_as_list")]
    pub locks: BTreeMap<KernelDependency, KernelLock>,
}

impl FromIterator<(KernelDependency, KernelLock)> for KernelLocks {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (KernelDependency, KernelLock)>,
    {
        Self {
            locks: iter.into_iter().collect(),
        }
    }
}

/// A locked kernel revision with the SRI hash of the Nix output path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct NixKernelLock {
    /// Locked git SHA.
    pub commit: Oid,

    /// SRI hash of the Nix store path.
    pub hash: String,
}

/// Multiple (Nix) kernel locks keyed by the dependency they resolve.
///
/// This data structure is used to store lock files to be consumed
/// by nix-builder.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct NixKernelLocks {
    #[serde(with = "dependency_map_as_list")]
    pub locks: BTreeMap<KernelDependency, NixKernelLock>,
}

impl FromIterator<(KernelDependency, NixKernelLock)> for NixKernelLocks {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (KernelDependency, NixKernelLock)>,
    {
        Self {
            locks: iter.into_iter().collect(),
        }
    }
}

/// A collection of kernel paths keyed by the dependency they resolve.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct KernelPaths {
    #[serde(with = "paths_as_list")]
    pub paths: BTreeMap<KernelDependency, PathBuf>,
}

impl FromIterator<(KernelDependency, PathBuf)> for KernelPaths {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (KernelDependency, PathBuf)>,
    {
        Self {
            paths: iter.into_iter().collect(),
        }
    }
}

/// Serialize `BTreeMap<KernelDependency, L>` as a list of
/// `{"dependency": ..., "lock": ...}` objects so that the dependency can be
/// preserved as a structured value while remaining valid JSON (JSON object
/// keys must be strings).
///
/// This is generic over the lock type so that [`KernelLocks`](super::KernelLocks)
/// and [`NixKernelLocks`](super::NixKernelLocks) share a single representation.
mod dependency_map_as_list {
    use std::collections::{BTreeMap, btree_map};

    use serde::{Deserialize, Serialize, de, ser};

    use crate::config::KernelDependency;

    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    struct EntryRef<'a, L> {
        dependency: &'a KernelDependency,
        lock: &'a L,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields, rename_all = "kebab-case")]
    struct Entry<L> {
        dependency: KernelDependency,
        lock: L,
    }

    pub fn serialize<S, L>(
        locks: &BTreeMap<KernelDependency, L>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
        L: Serialize,
    {
        serializer.collect_seq(
            locks
                .iter()
                .map(|(dependency, lock)| EntryRef { dependency, lock }),
        )
    }

    pub fn deserialize<'de, D, L>(
        deserializer: D,
    ) -> Result<BTreeMap<KernelDependency, L>, D::Error>
    where
        D: de::Deserializer<'de>,
        L: Deserialize<'de>,
    {
        let entries = Vec::<Entry<L>>::deserialize(deserializer)?;
        let mut locks = BTreeMap::new();
        for entry in entries {
            match locks.entry(entry.dependency) {
                btree_map::Entry::Vacant(slot) => {
                    slot.insert(entry.lock);
                }
                btree_map::Entry::Occupied(slot) => {
                    return Err(de::Error::custom(format!(
                        "duplicate dependency: {:?}",
                        slot.key()
                    )));
                }
            }
        }
        Ok(locks)
    }
}

/// Serialize `BTreeMap<KernelDependency, L>` as a list of
/// `{"dependency": ..., "path": ...}` objects so that the dependency can be
/// preserved as a structured value while remaining valid JSON (JSON object
/// keys must be strings).
mod paths_as_list {
    // Note, despite the similarities of this module to `dependency_map_as_list`,
    // we cannot make reuse that since the field names are different. Using a
    // macro probably only complicates the already dense serialization code.
    //
    // Basic rule: when in doubt, don't use macros.

    use std::collections::{BTreeMap, btree_map};
    use std::path::{Path, PathBuf};

    use serde::{Deserialize, Serialize, de, ser};

    use crate::config::KernelDependency;

    /// Borrowing counterpart of [`Entry`], so that serializing does not have to
    /// clone the paths.
    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    struct EntryRef<'a> {
        dependency: &'a KernelDependency,
        path: &'a Path,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields, rename_all = "kebab-case")]
    struct Entry {
        dependency: KernelDependency,
        path: PathBuf,
    }

    pub fn serialize<S>(
        paths: &BTreeMap<KernelDependency, PathBuf>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        serializer.collect_seq(
            paths
                .iter()
                .map(|(dependency, path)| EntryRef { dependency, path }),
        )
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<KernelDependency, PathBuf>, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let entries = Vec::<Entry>::deserialize(deserializer)?;
        let mut paths = BTreeMap::new();
        for entry in entries {
            match paths.entry(entry.dependency) {
                btree_map::Entry::Vacant(slot) => {
                    slot.insert(entry.path);
                }
                btree_map::Entry::Occupied(slot) => {
                    return Err(de::Error::custom(format!(
                        "duplicate dependency: {:?}",
                        slot.key()
                    )));
                }
            }
        }
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use serde_json::json;

    use crate::config::{KernelDependency, KernelVersion};
    use crate::git::Oid;

    use super::{KernelLock, KernelLocks, KernelPaths, NixKernelLock, NixKernelLocks};

    fn dependency(repo_id: &str, version: usize) -> KernelDependency {
        KernelDependency {
            repo_id: repo_id.to_string(),
            version: KernelVersion::Version(version),
        }
    }

    fn oid(commit: &str) -> Oid {
        Oid::from_str(commit).unwrap()
    }

    fn lock(commit: &str) -> KernelLock {
        KernelLock {
            commit: oid(commit),
        }
    }

    fn sample_locks() -> KernelLocks {
        KernelLocks::from_iter([
            (
                dependency("kernels-test/versions", 2),
                lock("f609e51b856b3d874b0ae8445913e200f02c1735"),
            ),
            (
                dependency("kernels-community/relu", 1),
                lock("d649efb56fb249ac8f7a57fa1866728ad0c60e52"),
            ),
        ])
    }

    #[test]
    fn kernel_lock_serializes_to_expected_json() {
        assert_eq!(
            serde_json::to_value(lock("d649efb56fb249ac8f7a57fa1866728ad0c60e52")).unwrap(),
            json!({"commit": "d649efb56fb249ac8f7a57fa1866728ad0c60e52"})
        );
    }

    #[test]
    fn kernel_lock_unknown_field_is_rejected() {
        let json = r#"{
            "commit": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "frobnicate": true
        }"#;
        assert!(serde_json::from_str::<KernelLock>(json).is_err());
    }

    #[test]
    fn kernel_locks_serializes_to_expected_json() {
        assert_eq!(
            serde_json::to_value(sample_locks()).unwrap(),
            json!([
                {
                    "dependency": {
                        "repo-id": "kernels-community/relu",
                        "version": 1,
                    },
                    "lock": {"commit": "d649efb56fb249ac8f7a57fa1866728ad0c60e52"},
                },
                {
                    "dependency": {
                        "repo-id": "kernels-test/versions",
                        "version": 2,
                    },
                    "lock": {"commit": "f609e51b856b3d874b0ae8445913e200f02c1735"},
                },
            ])
        );
    }

    #[test]
    fn kernel_locks_round_trips_through_json() {
        let locks = sample_locks();
        let json = serde_json::to_string(&locks).unwrap();
        assert_eq!(serde_json::from_str::<KernelLocks>(&json).unwrap(), locks);
    }

    #[test]
    fn kernel_locks_with_revision_dependency_round_trips() {
        let locks = KernelLocks::from_iter([(
            KernelDependency {
                repo_id: "kernels-test/versions".to_string(),
                version: KernelVersion::Revision("34fa".to_string()),
            },
            lock("f609e51b856b3d874b0ae8445913e200f02c1735"),
        )]);

        let json = serde_json::to_string(&locks).unwrap();
        assert_eq!(serde_json::from_str::<KernelLocks>(&json).unwrap(), locks);
    }

    #[test]
    fn locks_order_is_independent_of_insertion_order() {
        let forward = sample_locks();
        let reverse = forward
            .locks
            .iter()
            .rev()
            .map(|(dependency, lock)| (dependency.clone(), lock.clone()))
            .collect::<KernelLocks>();

        assert_eq!(
            serde_json::to_string(&forward).unwrap(),
            serde_json::to_string(&reverse).unwrap()
        );
    }

    #[test]
    fn kernel_locks_parses_from_json() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {"commit": "f609e51b856b3d874b0ae8445913e200f02c1735"}
            }
        ]"#;

        let locks: KernelLocks = serde_json::from_str(json).unwrap();
        assert_eq!(locks.locks.len(), 1);
        assert_eq!(
            locks.locks.get(&dependency("kernels-test/versions", 2)),
            Some(&lock("f609e51b856b3d874b0ae8445913e200f02c1735"))
        );
    }

    #[test]
    fn duplicate_lock_is_rejected() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {"commit": "f609e51b856b3d874b0ae8445913e200f02c1735"}
            },
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {"commit": "1111111111111111111111111111111111111111"}
            }
        ]"#;

        let err = serde_json::from_str::<KernelLocks>(json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate dependency"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nix_kernel_locks_only_add_a_hash_to_each_lock() {
        let locks = sample_locks();
        let nix_locks = locks
            .locks
            .iter()
            .map(|(dependency, lock)| {
                (
                    dependency.clone(),
                    NixKernelLock {
                        commit: lock.commit.clone(),
                        hash: "sha256-1CM0MGEOCqqnYV989jcUANEbiB727JftjXTdQVUHPwk=".to_string(),
                    },
                )
            })
            .collect::<NixKernelLocks>();

        let mut expected = serde_json::to_value(&locks).unwrap();
        for entry in expected.as_array_mut().unwrap() {
            entry["lock"]["hash"] = json!("sha256-1CM0MGEOCqqnYV989jcUANEbiB727JftjXTdQVUHPwk=");
        }

        assert_eq!(serde_json::to_value(&nix_locks).unwrap(), expected);
    }

    #[test]
    fn nix_kernel_locks_round_trip_through_json() {
        let locks = NixKernelLocks::from_iter([(
            dependency("kernels-community/relu", 1),
            NixKernelLock {
                commit: oid("d649efb56fb249ac8f7a57fa1866728ad0c60e52"),
                hash: "sha256-1CM0MGEOCqqnYV989jcUANEbiB727JftjXTdQVUHPwk=".to_string(),
            },
        )]);

        let json = serde_json::to_string(&locks).unwrap();
        assert_eq!(
            serde_json::from_str::<NixKernelLocks>(&json).unwrap(),
            locks
        );
    }

    #[test]
    fn nix_kernel_lock_without_hash_is_rejected() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {"commit": "f609e51b856b3d874b0ae8445913e200f02c1735"}
            }
        ]"#;

        assert!(serde_json::from_str::<NixKernelLocks>(json).is_err());
    }

    #[test]
    fn lock_with_invalid_commit_is_rejected() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {"commit": "f609e51"}
            }
        ]"#;

        assert!(serde_json::from_str::<KernelLocks>(json).is_err());
    }

    #[test]
    fn locks_as_object_is_rejected() {
        assert!(serde_json::from_str::<KernelLocks>("{}").is_err());
    }

    #[test]
    fn lock_entry_without_lock_is_rejected() {
        let json = r#"[
            {"dependency": {"repo-id": "kernels-test/versions", "version": 2}}
        ]"#;
        assert!(serde_json::from_str::<KernelLocks>(json).is_err());
    }

    fn sample_paths() -> KernelPaths {
        KernelPaths::from_iter([
            (
                dependency("kernels-test/versions", 2),
                PathBuf::from("/kernels/versions"),
            ),
            (
                dependency("kernels-community/relu", 1),
                PathBuf::from("/kernels/relu"),
            ),
        ])
    }

    #[test]
    fn kernel_paths_serialize_to_expected_json() {
        assert_eq!(
            serde_json::to_value(sample_paths()).unwrap(),
            json!([
                {
                    "dependency": {"repo-id": "kernels-community/relu", "version": 1},
                    "path": "/kernels/relu",
                },
                {
                    "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                    "path": "/kernels/versions",
                },
            ])
        );
    }

    #[test]
    fn kernel_paths_round_trip_through_json() {
        let paths = sample_paths();
        let json = serde_json::to_string(&paths).unwrap();
        assert_eq!(serde_json::from_str::<KernelPaths>(&json).unwrap(), paths);
    }

    #[test]
    fn duplicate_path_is_rejected() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "path": "/kernels/versions"
            },
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "path": "/kernels/versions-2"
            }
        ]"#;

        let err = serde_json::from_str::<KernelPaths>(json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate dependency"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn path_entry_without_path_is_rejected() {
        let json = r#"[
            {"dependency": {"repo-id": "kernels-test/versions", "version": 2}}
        ]"#;
        assert!(serde_json::from_str::<KernelPaths>(json).is_err());
    }
}
