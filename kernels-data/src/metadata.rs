use std::str::FromStr;

use eyre::Result;
use serde::{Deserialize, Serialize};

use crate::config::{Backend, Build, GitUrl, KernelName};
use crate::digest::Digest;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct BackendInfo {
    #[serde(rename = "type")]
    pub backend_type: Backend,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archs: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct GitHash {
    pub sha: String,
    pub dirty: bool,
}

/// Provenance of the `kernel-builder` that produced a build.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct KernelBuilderVersion {
    pub version: String,
    /// Commit SHA + dirty state of the `kernel-builder` source, when known.
    #[serde(flatten)]
    pub git: Option<GitHash>,
}

/// Provenance of a kernel build: the git state of the `kernel-builder` and of
/// the kernel source it was built from.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_builder: Option<KernelBuilderVersion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel: Option<GitHash>,
}

impl Provenance {
    /// Whether either the `kernel-builder` or the kernel source was dirty.
    pub fn is_dirty(&self) -> bool {
        self.kernel_builder
            .as_ref()
            .and_then(|kb| kb.git.as_ref())
            .is_some_and(|g| g.dirty)
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
        let archs = build.framework.precomputable_backend_archs(backend);

        Ok(Self {
            id,
            name: build.general.name.clone(),
            version: build.general.version,
            license: build.general.license.clone(),
            upstream: build.general.upstream.clone(),
            source: build.general.source.clone(),
            python_depends,
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

    use crate::config::{Backend, Build, Framework, General, KernelName, TorchNoarch, TvmFfi};

    use super::{GitHash, KernelBuilderVersion, Metadata, Provenance};

    fn sample_provenance(kernel_builder_dirty: bool, kernel_dirty: bool) -> Provenance {
        Provenance {
            kernel_builder: Some(KernelBuilderVersion {
                version: "0.1.0".to_string(),
                git: Some(GitHash {
                    sha: "a".repeat(40),
                    dirty: kernel_builder_dirty,
                }),
            }),
            kernel: Some(GitHash {
                sha: "b".repeat(40),
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
                python_depends: None,
                cuda: None,
                neuron: None,
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
                python_depends: None,
                cuda: None,
                neuron: None,
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
        assert_eq!(provenance.kernel.unwrap().sha, "b".repeat(40));

        let kernel_builder = provenance.kernel_builder.unwrap();
        assert_eq!(kernel_builder.version, "0.1.0");
        // The embedded `GitHash` is flattened into the `kernel-builder` object.
        let git = kernel_builder
            .git
            .expect("kernel-builder git should round-trip");
        assert_eq!(git.sha, "a".repeat(40));
        assert!(!git.dirty);
    }

    #[test]
    fn kernel_builder_version_flattens_git_hash() {
        let kernel_builder = KernelBuilderVersion {
            version: "0.1.0".to_string(),
            git: Some(GitHash {
                sha: "c".repeat(40),
                dirty: true,
            }),
        };

        // The `GitHash` fields are flattened into the same object as `version`.
        let value: serde_json::Value = serde_json::to_value(&kernel_builder).unwrap();
        assert_eq!(value["version"], "0.1.0");
        assert_eq!(value["sha"], "c".repeat(40));
        assert_eq!(value["dirty"], true);
        assert!(value.get("git").is_none());

        let parsed: KernelBuilderVersion = serde_json::from_value(value).unwrap();
        let git = parsed.git.expect("git should round-trip");
        assert_eq!(git.sha, "c".repeat(40));
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
}
