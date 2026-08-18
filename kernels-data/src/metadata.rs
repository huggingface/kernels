use std::str::FromStr;

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::config::{Backend, Build, GitUrl, KernelDependency, KernelName};
use crate::digest::Digest;
use crate::git::GitStatus;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archs: Option<Vec<String>>,
}

/// Provenance of the `kernel-builder` that produced a build.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct KernelBuilderVersion {
    pub version: String,
    /// Git state of the `kernel-builder` source, when known.
    ///
    /// Flattened so that the `kernel-builder` and kernel provenance have the
    /// same shape in the serialized form.
    #[serde(flatten)]
    pub git: Option<GitStatus>,
}

/// Provenance of a kernel build: the git state of the `kernel-builder` and of
/// the kernel source it was built from.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Provenance {
    /// The `kernel-builder` that produced the build. Always known: it is baked
    /// into the `kernel-builder` binary at compile time.
    pub kernel_builder: KernelBuilderVersion,
    /// Git provenance of the kernel source. `None` when the kernel source was
    /// not built from a git repository (so its revision is unknown).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<GitStatus>,
}

impl Provenance {
    /// Whether either the `kernel-builder` or the kernel source was dirty.
    pub fn is_dirty(&self) -> bool {
        self.kernel_builder.git.as_ref().is_some_and(|g| g.dirty)
            || self.kernel.as_ref().is_some_and(|k| k.dirty)
    }
}

/// Kernel metadata.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Metadata {
    pub name: KernelName,
    pub id: String,
    pub version: usize,
    pub license: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<GitUrl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<GitUrl>,
    pub python_depends: Vec<String>,
    #[serde(default)]
    pub kernel_depends: Vec<KernelDependency>,
    pub backend: BackendInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<Digest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

impl Metadata {
    /// Construct metadata for a specific backend.
    ///
    /// This constructor creates metadata for a specific backend from the
    /// kernel build configuration and kernel identifier.
    ///
    /// Supported backend archs are only supported for Torch noarch, since
    /// the archs need to be computed at build time for arch frameworks.
    pub fn for_backend(build: &Build, id: String, backend: Backend) -> Result<Self> {
        let python_depends = build.general.all_python_depends(backend)?;
        let kernel_depends = build.general.all_kernel_depends(backend);
        let archs = build.framework.precomputable_backend_archs(backend);

        Ok(Self {
            id,
            name: build.general.name.clone(),
            version: build.general.version,
            license: build.general.license.clone(),
            upstream: build.general.upstream.clone(),
            source: build.general.source.clone(),
            python_depends,
            kernel_depends,
            backend: BackendInfo {
                archs,
                backend_type: backend,
            },
            digest: None,
            provenance: None,
        })
    }

    /// Read the metadata from a JSON byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Read the metadata from a JSON `std::io::Read`.
    pub fn from_reader<R: std::io::Read>(reader: R) -> Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }
}

impl FromStr for Metadata {
    type Err = eyre::Report;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::str::FromStr;

    use crate::config::{Backend, Build, Framework, General, KernelName, TorchNoarch, TvmFfi};
    use crate::git::{GitStatus, Oid};

    use super::{KernelBuilderVersion, Metadata, Provenance};

    /// During the development of kernels 0.17, we had a time period where
    /// `sha` was used before `oid`. The Hub as a bunch of kernels with `sha`
    /// in their metadata. For this reason, we have a compatibility alias.
    /// This older metadata is used to validate that metadata with `sha` can
    /// be parsed correctly.
    const LEGACY_HUB_METADATA: &str = include_str!("../tests/data/legacy-sha-provenance.json");

    fn oid(hex_digit: &str) -> Oid {
        Oid::from_str(&hex_digit.repeat(40)).unwrap()
    }

    fn sample_provenance(kernel_builder_dirty: bool, kernel_dirty: bool) -> Provenance {
        Provenance {
            kernel_builder: KernelBuilderVersion {
                version: "0.1.0".to_string(),
                git: Some(GitStatus {
                    commit: oid("a"),
                    dirty: kernel_builder_dirty,
                }),
            },
            kernel: Some(GitStatus {
                commit: oid("b"),
                dirty: kernel_dirty,
            }),
        }
    }

    fn torch_noarch_build() -> Build {
        Build {
            general: General {
                name: KernelName::new("test-kernel").unwrap(),
                version: 1,
                license: "apache-2.0".to_string(),
                upstream: None,
                source: None,
                backends: vec![Backend::Cuda, Backend::Rocm, Backend::Cpu],
                hub: None,
                kernel_depends: None,
                python_depends: None,
                cuda: None,
                neuron: None,
                tpu: None,
                xpu: None,
            },
            kernels: HashMap::new(),
            framework: Framework::TorchNoarch(TorchNoarch {
                pyext: None,
                cuda_capabilities: Some(vec!["7.0".to_string(), "8.0".to_string()]),
                rocm_archs: Some(vec!["gfx90a".to_string()]),
            }),
        }
    }

    #[test]
    fn cuda_archs_for_torch_noarch() {
        let build = torch_noarch_build();
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Cuda).unwrap();

        assert_eq!(metadata.backend.backend_type, Backend::Cuda);
        assert_eq!(
            metadata.backend.archs,
            Some(vec!["7.0".to_string(), "8.0".to_string()])
        );
    }

    #[test]
    fn rocm_archs_for_torch_noarch() {
        let build = torch_noarch_build();
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Rocm).unwrap();

        assert_eq!(metadata.backend.backend_type, Backend::Rocm);
        assert_eq!(metadata.backend.archs, Some(vec!["gfx90a".to_string()]));
    }

    #[test]
    fn no_archs_for_cpu_with_torch_noarch() {
        let build = torch_noarch_build();
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Cpu).unwrap();

        assert_eq!(metadata.backend.backend_type, Backend::Cpu);
        assert!(metadata.backend.archs.is_none());
    }

    #[test]
    fn no_archs_for_arch_framework() {
        let build = Build {
            general: General {
                name: KernelName::new("test-kernel").unwrap(),
                version: 1,
                license: "apache-2.0".to_string(),
                upstream: None,
                source: None,
                backends: vec![Backend::Cuda],
                hub: None,
                kernel_depends: None,
                python_depends: None,
                cuda: None,
                neuron: None,
                tpu: None,
                xpu: None,
            },
            kernels: HashMap::new(),
            framework: Framework::TvmFfi(TvmFfi {
                include: None,
                pyext: None,
                src: vec![],
                cxx_flags: None,
            }),
        };
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Cuda).unwrap();

        assert_eq!(metadata.backend.backend_type, Backend::Cuda);
        assert!(metadata.backend.archs.is_none());
    }

    #[test]
    fn provenance_is_dirty_reflects_either_source() {
        assert!(!sample_provenance(false, false).is_dirty());
        assert!(sample_provenance(true, false).is_dirty());
        assert!(sample_provenance(false, true).is_dirty());
        assert!(sample_provenance(true, true).is_dirty());
    }

    #[test]
    fn metadata_without_provenance_serializes_without_key() {
        let build = torch_noarch_build();
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Cuda).unwrap();

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(!json.contains("provenance"));

        let parsed: Metadata = serde_json::from_str(&json).unwrap();
        assert!(parsed.provenance.is_none());
    }

    #[test]
    fn metadata_always_emits_kernel_depends() {
        let build = torch_noarch_build();
        let metadata = Metadata::for_backend(&build, "test-id".to_string(), Backend::Cuda).unwrap();

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("\"kernel-depends\":[]"));

        let parsed: Metadata = serde_json::from_str(&json).unwrap();
        assert!(parsed.kernel_depends.is_empty());
    }

    #[test]
    fn metadata_round_trips_provenance_in_kebab_case() {
        let build = torch_noarch_build();
        let mut metadata =
            Metadata::for_backend(&build, "test-id".to_string(), Backend::Cuda).unwrap();
        metadata.provenance = Some(sample_provenance(false, true));

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("\"provenance\""));
        assert!(json.contains("\"kernel-builder\""));

        let parsed: Metadata = serde_json::from_str(&json).unwrap();
        let provenance = parsed.provenance.expect("provenance should round-trip");
        assert!(provenance.is_dirty());
        assert_eq!(provenance.kernel.unwrap().commit, oid("b"));

        let kernel_builder = provenance.kernel_builder;
        assert_eq!(kernel_builder.version, "0.1.0");
        // The embedded `GitStatus` is flattened into the `kernel-builder` object.
        let git = kernel_builder
            .git
            .expect("kernel-builder git should round-trip");
        assert_eq!(git.commit, oid("a"));
        assert!(!git.dirty);
    }

    #[test]
    fn kernel_builder_version_flattens_git_status() {
        let kernel_builder = KernelBuilderVersion {
            version: "0.1.0".to_string(),
            git: Some(GitStatus {
                commit: oid("c"),
                dirty: true,
            }),
        };

        // The `GitStatus` fields are flattened into the same object as `version`.
        let value: serde_json::Value = serde_json::to_value(&kernel_builder).unwrap();
        assert_eq!(value["version"], "0.1.0");
        assert_eq!(value["commit"], "c".repeat(40));
        assert_eq!(value["dirty"], true);
        assert!(value.get("git").is_none());

        let parsed: KernelBuilderVersion = serde_json::from_value(value).unwrap();
        let git = parsed.git.expect("git should round-trip");
        assert_eq!(git.commit, oid("c"));
        assert!(git.dirty);
    }

    #[test]
    fn kernel_builder_version_without_git_round_trips_to_none() {
        // When the `kernel-builder` git provenance is unknown, only `version`
        // is serialized and it must deserialize back to `git: None`.
        let kernel_builder = KernelBuilderVersion {
            version: "0.1.0".to_string(),
            git: None,
        };

        let json = serde_json::to_string(&kernel_builder).unwrap();
        assert_eq!(json, r#"{"version":"0.1.0"}"#);

        let parsed: KernelBuilderVersion = serde_json::from_str(&json).unwrap();
        assert!(parsed.git.is_none());
    }

    #[test]
    fn legacy_hub_metadata_still_parses() {
        let metadata = Metadata::from_str(LEGACY_HUB_METADATA).unwrap();

        // Provenance is read through the `sha` alias rather than discarded.
        let provenance = metadata.provenance.expect("provenance should be read");
        assert!(!provenance.is_dirty());

        let kernel_builder = provenance.kernel_builder;
        assert_eq!(kernel_builder.version, "0.17.0-dev0");
        let git = kernel_builder
            .git
            .expect("kernel-builder git should be read");
        assert_eq!(
            git.commit,
            Oid::from_str("d0610aa58db33b142c86b59598a2a1c730f52996").unwrap()
        );
        assert!(!git.dirty);

        let kernel = provenance.kernel.expect("kernel git should be read");
        assert_eq!(
            kernel.commit,
            Oid::from_str("d45addadb0c380f3d7cc2813310b3bcd75f8aa9a").unwrap()
        );
        assert!(!kernel.dirty);

        // The rest of the metadata is unaffected.
        assert_eq!(metadata.name.as_ref(), "einops");
        assert_eq!(metadata.id, "_einops_cpu_d45adda");
        assert_eq!(metadata.version, 1);
        assert_eq!(metadata.license, "MIT");
        assert!(metadata.upstream.is_some());
        assert!(metadata.source.is_none());
        assert!(metadata.python_depends.is_empty());
        assert!(metadata.kernel_depends.is_empty());
        assert_eq!(metadata.backend.backend_type, Backend::Cpu);
        assert!(metadata.digest.is_some());
    }

    #[test]
    fn legacy_hub_metadata_is_reserialized_with_commit() {
        let metadata = Metadata::from_str(LEGACY_HUB_METADATA).unwrap();
        let json = serde_json::to_string(&metadata).unwrap();

        assert!(json.contains(r#""commit":"d0610aa58db33b142c86b59598a2a1c730f52996""#));
        assert!(json.contains(r#""commit":"d45addadb0c380f3d7cc2813310b3bcd75f8aa9a""#));
        assert!(!json.contains("\"sha\""));
    }

    #[test]
    fn metadata_from_newer_writer_ignores_unknown_fields() {
        // Simulate metadata written by a newer `kernel-builder` that added
        // fields we do not know about.
        let mut value: serde_json::Value = serde_json::from_str(LEGACY_HUB_METADATA).unwrap();
        value["future-field"] = serde_json::json!("surprise");
        value["provenance"]["future-field"] = serde_json::json!({"nested": [1, 2, 3]});
        value["provenance"]["kernel"]["future-field"] = serde_json::json!(true);

        let metadata: Metadata = serde_json::from_value(value).unwrap();
        let provenance = metadata.provenance.expect("provenance should be read");
        assert_eq!(
            provenance.kernel.unwrap().commit,
            Oid::from_str("d45addadb0c380f3d7cc2813310b3bcd75f8aa9a").unwrap()
        );
    }

    #[test]
    fn unreadable_provenance_is_an_error() {
        for provenance in [
            serde_json::json!(42),
            serde_json::json!({}),
            serde_json::json!({"kernel-builder": {"version": "0.1.0"}, "kernel": "nonsense"}),
            serde_json::json!({
                "kernel-builder": {"version": "0.1.0"},
                "kernel": {"commit": "not-an-oid", "dirty": false},
            }),
        ] {
            let mut value: serde_json::Value = serde_json::from_str(LEGACY_HUB_METADATA).unwrap();
            value["provenance"] = provenance.clone();

            assert!(
                serde_json::from_value::<Metadata>(value).is_err(),
                "should be rejected: {provenance}"
            );
        }
    }

    #[test]
    fn unreadable_kernel_builder_git_is_dropped() {
        // `KernelBuilderVersion::git` is flattened, and serde deserializes a
        // flattened `Option` by discarding the error rather than propagating
        // it. So an unreadable `kernel-builder` git status drops that status
        // instead of rejecting the metadata.
        let mut value: serde_json::Value = serde_json::from_str(LEGACY_HUB_METADATA).unwrap();
        value["provenance"]["kernel-builder"]["sha"] = serde_json::json!("not-an-oid");

        let metadata: Metadata = serde_json::from_value(value).unwrap();
        let provenance = metadata.provenance.expect("provenance should be read");
        assert!(provenance.kernel_builder.git.is_none());
        assert_eq!(provenance.kernel_builder.version, "0.17.0-dev0");
        assert!(provenance.kernel.is_some());
    }

    #[test]
    fn unreadable_digest_is_an_error() {
        let mut value: serde_json::Value = serde_json::from_str(LEGACY_HUB_METADATA).unwrap();
        value["digest"]["algorithm"] = serde_json::json!("sha3-512");

        assert!(serde_json::from_value::<Metadata>(value).is_err());
    }
}
