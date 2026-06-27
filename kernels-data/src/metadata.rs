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

    use super::Metadata;

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
}
