use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::KernelDependency;

/// A locked kernel revision and its transitive dependencies.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct KernelLock {
    pub repo_id: String,
    pub revision: String,
    pub depends: KernelLocks,
}

/// A collection of locked kernels keyed by the dependency they resolve.
///
/// This is a transparent wrapper around the map: it serializes as the list of
/// locks itself, without an intermediate object.
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

/// Serialize `BTreeMap<KernelDependency, KernelLock>` as a list of
/// `{"dependency": ..., "lock": ...}` objects so that the dependency can be
/// preserved as a structured value while remaining valid JSON (JSON object
/// keys must be strings).
///
/// The map is ordered, so the list order only depends on the locked
/// dependencies themselves, never on insertion or hashing order.
mod dependency_map_as_list {
    use std::collections::{BTreeMap, btree_map};

    use serde::{Deserialize, Serialize, de, ser};

    use super::KernelLock;
    use crate::config::KernelDependency;

    /// Borrowing counterpart of [`Entry`], so that serializing does not have to
    /// clone the (potentially deep) dependency tree.
    #[derive(Serialize)]
    #[serde(rename_all = "kebab-case")]
    struct EntryRef<'a> {
        dependency: &'a KernelDependency,
        lock: &'a KernelLock,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields, rename_all = "kebab-case")]
    struct Entry {
        dependency: KernelDependency,
        lock: KernelLock,
    }

    pub fn serialize<S>(
        locks: &BTreeMap<KernelDependency, KernelLock>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        serializer.collect_seq(
            locks
                .iter()
                .map(|(dependency, lock)| EntryRef { dependency, lock }),
        )
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<KernelDependency, KernelLock>, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let entries = Vec::<Entry>::deserialize(deserializer)?;
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::config::{KernelDependency, KernelVersion};

    use super::{KernelLock, KernelLocks};

    fn dependency(repo_id: &str, version: usize) -> KernelDependency {
        KernelDependency {
            repo_id: repo_id.to_string(),
            version: KernelVersion::Version(version),
        }
    }

    fn leaf_lock(repo_id: &str, revision: &str) -> KernelLock {
        KernelLock {
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            depends: KernelLocks::default(),
        }
    }

    /// A lock with two direct dependencies, one of which has a dependency of
    /// its own, so that the nested (recursive) representation is covered.
    fn nested_lock() -> KernelLock {
        let inner_depends = KernelLocks::from_iter([(
            dependency("kernels-test/nested", 3),
            leaf_lock(
                "kernels-test/nested",
                "1111111111111111111111111111111111111111",
            ),
        )]);

        let depends = KernelLocks::from_iter([
            (
                dependency("kernels-test/versions", 2),
                KernelLock {
                    repo_id: "kernels-test/versions".to_string(),
                    revision: "f609e51b856b3d874b0ae8445913e200f02c1735".to_string(),
                    depends: inner_depends,
                },
            ),
            (
                dependency("kernels-test/activation", 1),
                leaf_lock(
                    "kernels-test/activation",
                    "2222222222222222222222222222222222222222",
                ),
            ),
        ]);

        KernelLock {
            repo_id: "kernels-community/relu".to_string(),
            revision: "d649efb56fb249ac8f7a57fa1866728ad0c60e52".to_string(),
            depends,
        }
    }

    fn sample_locks() -> KernelLocks {
        KernelLocks::from_iter([
            (dependency("kernels-community/relu", 1), nested_lock()),
            (
                dependency("kernels-test/activation", 1),
                leaf_lock(
                    "kernels-test/activation",
                    "2222222222222222222222222222222222222222",
                ),
            ),
        ])
    }

    #[test]
    fn kernel_lock_serializes_to_expected_json() {
        assert_eq!(
            serde_json::to_value(nested_lock()).unwrap(),
            json!({
                "repo-id": "kernels-community/relu",
                "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
                "depends": [
                    {
                        "dependency": {
                            "repo-id": "kernels-test/activation",
                            "version": 1,
                        },
                        "lock": {
                            "repo-id": "kernels-test/activation",
                            "revision": "2222222222222222222222222222222222222222",
                            "depends": [],
                        },
                    },
                    {
                        "dependency": {
                            "repo-id": "kernels-test/versions",
                            "version": 2,
                        },
                        "lock": {
                            "repo-id": "kernels-test/versions",
                            "revision": "f609e51b856b3d874b0ae8445913e200f02c1735",
                            "depends": [
                                {
                                    "dependency": {
                                        "repo-id": "kernels-test/nested",
                                        "version": 3,
                                    },
                                    "lock": {
                                        "repo-id": "kernels-test/nested",
                                        "revision": "1111111111111111111111111111111111111111",
                                        "depends": [],
                                    },
                                },
                            ],
                        },
                    },
                ],
            })
        );
    }

    #[test]
    fn kernel_lock_round_trips_through_json() {
        let lock = nested_lock();
        let json = serde_json::to_string(&lock).unwrap();
        assert_eq!(serde_json::from_str::<KernelLock>(&json).unwrap(), lock);
    }

    #[test]
    fn kernel_lock_with_revision_dependency_round_trips() {
        let lock = KernelLock {
            repo_id: "kernels-community/relu".to_string(),
            revision: "d649efb56fb249ac8f7a57fa1866728ad0c60e52".to_string(),
            depends: KernelLocks::from_iter([(
                KernelDependency {
                    repo_id: "kernels-test/versions".to_string(),
                    version: KernelVersion::Revision("34fa".to_string()),
                },
                leaf_lock(
                    "kernels-test/versions",
                    "f609e51b856b3d874b0ae8445913e200f02c1735",
                ),
            )]),
        };

        let json = serde_json::to_string(&lock).unwrap();
        assert_eq!(serde_json::from_str::<KernelLock>(&json).unwrap(), lock);
    }

    #[test]
    fn depends_order_is_independent_of_insertion_order() {
        let forward = nested_lock();
        let mut reverse = forward.clone();
        reverse.depends = forward
            .depends
            .locks
            .iter()
            .rev()
            .map(|(dependency, lock)| (dependency.clone(), lock.clone()))
            .collect();

        assert_eq!(
            serde_json::to_string(&forward).unwrap(),
            serde_json::to_string(&reverse).unwrap()
        );
    }

    #[test]
    fn kernel_lock_parses_from_json() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "depends": [
                {
                    "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                    "lock": {
                        "repo-id": "kernels-test/versions",
                        "revision": "f609e51b856b3d874b0ae8445913e200f02c1735",
                        "depends": []
                    }
                }
            ]
        }"#;

        let lock: KernelLock = serde_json::from_str(json).unwrap();
        assert_eq!(lock.repo_id, "kernels-community/relu");
        assert_eq!(lock.revision, "d649efb56fb249ac8f7a57fa1866728ad0c60e52");
        assert_eq!(
            lock.depends
                .locks
                .get(&dependency("kernels-test/versions", 2)),
            Some(&leaf_lock(
                "kernels-test/versions",
                "f609e51b856b3d874b0ae8445913e200f02c1735"
            ))
        );
    }

    #[test]
    fn duplicate_dependency_is_rejected() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "depends": [
                {
                    "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                    "lock": {
                        "repo-id": "kernels-test/versions",
                        "revision": "f609e51b856b3d874b0ae8445913e200f02c1735",
                        "depends": []
                    }
                },
                {
                    "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                    "lock": {
                        "repo-id": "kernels-test/versions",
                        "revision": "1111111111111111111111111111111111111111",
                        "depends": []
                    }
                }
            ]
        }"#;

        let err = serde_json::from_str::<KernelLock>(json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate dependency"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn depends_as_object_is_rejected() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "depends": {}
        }"#;
        assert!(serde_json::from_str::<KernelLock>(json).is_err());
    }

    #[test]
    fn depends_entry_without_lock_is_rejected() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "depends": [
                {"dependency": {"repo-id": "kernels-test/versions", "version": 2}}
            ]
        }"#;
        assert!(serde_json::from_str::<KernelLock>(json).is_err());
    }

    #[test]
    fn missing_depends_is_rejected() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52"
        }"#;
        assert!(serde_json::from_str::<KernelLock>(json).is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let json = r#"{
            "repo-id": "kernels-community/relu",
            "revision": "d649efb56fb249ac8f7a57fa1866728ad0c60e52",
            "depends": [],
            "frobnicate": true
        }"#;
        assert!(serde_json::from_str::<KernelLock>(json).is_err());
    }

    #[test]
    fn kernel_locks_round_trips_through_json() {
        let locks = sample_locks();
        let json = serde_json::to_string(&locks).unwrap();
        assert_eq!(serde_json::from_str::<KernelLocks>(&json).unwrap(), locks);
    }

    #[test]
    fn kernel_locks_serializes_as_bare_list() {
        let value = serde_json::to_value(sample_locks()).unwrap();
        let locks = value.as_array().expect("locks should serialize as a list");

        assert_eq!(locks.len(), 2);
        assert_eq!(
            locks[0].get("dependency").unwrap(),
            &json!({"repo-id": "kernels-community/relu", "version": 1})
        );
        assert_eq!(
            locks[1].get("dependency").unwrap(),
            &json!({"repo-id": "kernels-test/activation", "version": 1})
        );
        assert_eq!(
            locks[1].get("lock").unwrap(),
            &json!({
                "repo-id": "kernels-test/activation",
                "revision": "2222222222222222222222222222222222222222",
                "depends": [],
            })
        );
    }

    #[test]
    fn duplicate_lock_is_rejected() {
        let json = r#"[
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {
                    "repo-id": "kernels-test/versions",
                    "revision": "f609e51b856b3d874b0ae8445913e200f02c1735",
                    "depends": []
                }
            },
            {
                "dependency": {"repo-id": "kernels-test/versions", "version": 2},
                "lock": {
                    "repo-id": "kernels-test/versions",
                    "revision": "1111111111111111111111111111111111111111",
                    "depends": []
                }
            }
        ]"#;

        let err = serde_json::from_str::<KernelLocks>(json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate dependency"),
            "unexpected error: {err}"
        );
    }
}
