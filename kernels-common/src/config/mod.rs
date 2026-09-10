use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
    path::{Path, PathBuf},
    str::FromStr,
};

use eyre::Result;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod deps;
pub use deps::{Dependency, PythonDependency};

mod compat;
pub use compat::BuildCompat;

mod git_url;
pub use git_url::GitUrl;

mod kernel_deps;
pub use kernel_deps::{KernelDependency, KernelVersion};

mod name;
pub use name::KernelName;

mod parse;

pub mod v3;
pub mod v4;
pub mod v5;

use itertools::Itertools;

use crate::version::Version;

pub type CurrentConfig = v5::Build;
pub const CURRENT_EDITION: usize = 5;

/// Baseline `kernels` version that can load kernels built with the current
/// metadata format.
const KERNELS_VERSION_BASELINE: Version<3> = Version::new([0, 14, 0]);

/// First `kernels` version that can resolve kernel dependencies.
const KERNELS_VERSION_KERNEL_DEPENDS: Version<3> = Version::new([0, 17, 0]);

pub struct Build {
    pub general: General,
    pub kernels: HashMap<String, Kernel>,
    pub framework: Framework,
}

impl Build {
    pub fn open(kernel_dir: impl AsRef<Path>) -> Result<Build> {
        let build_compat = parse::parse_and_validate(kernel_dir)?;
        Ok(build_compat.into())
    }

    pub fn is_noarch(&self) -> bool {
        matches!(self.framework, Framework::TorchNoarch(_))
    }

    pub fn branch(&self) -> Option<&str> {
        self.general.hub.as_ref().and_then(|h| h.branch.as_deref())
    }

    pub fn repo_id(&self) -> Option<&str> {
        self.general.hub.as_ref().and_then(|h| h.repo_id.as_deref())
    }

    /// Minimum version of the `kernels` Python library required to load a
    /// build variant for `backend`.
    ///
    /// The version is derived from the features that the kernel uses, rather
    /// than being specified by the kernel author, who would have to keep track
    /// of which `kernels` version introduced support for which feature.
    pub fn required_kernels_version(&self, backend: Backend) -> Version<3> {
        let mut required = KERNELS_VERSION_BASELINE;

        // Kernel dependencies are only fetched by newer versions of `kernels`,
        // older versions would fail to import the kernel.
        if !self.general.all_kernel_depends(backend).is_empty() {
            required = required.max(KERNELS_VERSION_KERNEL_DEPENDS);
        }

        required
    }
}

pub enum Framework {
    Torch(Torch),
    TorchNoarch(TorchNoarch),
    TvmFfi(TvmFfi),
}

impl Framework {
    pub fn torch(&self) -> Option<&Torch> {
        match self {
            Framework::Torch(torch) => Some(torch),
            _ => None,
        }
    }

    pub fn torch_noarch(&self) -> Option<&TorchNoarch> {
        match self {
            Framework::TorchNoarch(torch_noarch) => Some(torch_noarch),
            _ => None,
        }
    }

    pub fn tvm_ffi(&self) -> Option<&TvmFfi> {
        match self {
            Framework::TvmFfi(tvm_ffi) => Some(tvm_ffi),
            _ => None,
        }
    }

    pub(crate) fn precomputable_backend_archs(&self, backend: Backend) -> Option<Vec<String>> {
        match self {
            Framework::TorchNoarch(torch_noarch) => match backend {
                Backend::Cuda => torch_noarch.cuda_capabilities.clone(),
                Backend::Rocm => torch_noarch.rocm_archs.clone(),
                _ => None,
            },
            _ => None,
        }
    }
}

pub struct General {
    pub name: KernelName,

    /// Kernel API/ABI version.
    pub version: usize,

    /// Hugging Face Hub license identifier.
    pub license: String,

    /// Original upstream repository for the kernel code.
    pub upstream: Option<GitUrl>,

    /// Kernel-builder formatted source repository (must contain build.toml and flake.nix).
    pub source: Option<GitUrl>,

    pub backends: Vec<Backend>,
    pub hub: Option<Hub>,
    pub kernel_depends: Option<Vec<KernelDependency>>,
    pub python_depends: Option<Vec<String>>,

    pub cuda: Option<CudaGeneral>,
    pub neuron: Option<NeuronGeneral>,
    pub tpu: Option<TpuGeneral>,
    pub xpu: Option<XpuGeneral>,
}

impl General {
    pub fn general_python_depends(
        &self,
    ) -> Box<dyn Iterator<Item = Result<(&str, &PythonDependency)>> + '_> {
        let general_python_deps = match self.python_depends.as_ref() {
            Some(deps) => deps,
            None => {
                return Box::new(std::iter::empty());
            }
        };

        Box::new(general_python_deps.iter().map(move |dep| {
            match deps::PYTHON_DEPENDENCIES.get_dependency(dep) {
                Ok(resolved_deps) => Ok((dep.as_str(), resolved_deps)),
                Err(e) => Err(e.into()),
            }
        }))
    }

    pub fn backend_python_depends(
        &self,
        backend: Backend,
    ) -> Box<dyn Iterator<Item = Result<(&str, &PythonDependency)>> + '_> {
        let backend_python_deps = match backend {
            Backend::Cuda => self
                .cuda
                .as_ref()
                .and_then(|cuda| cuda.python_depends.as_ref()),
            Backend::Tpu => self
                .tpu
                .as_ref()
                .and_then(|tpu| tpu.python_depends.as_ref()),
            Backend::Xpu => self
                .xpu
                .as_ref()
                .and_then(|xpu| xpu.python_depends.as_ref()),
            _ => None,
        };

        let backend_python_deps = match backend_python_deps {
            Some(deps) => deps,
            None => {
                return Box::new(std::iter::empty());
            }
        };

        Box::new(backend_python_deps.iter().map(move |dep| {
            match deps::PYTHON_DEPENDENCIES.get_backend_dependency(backend, dep) {
                Ok(resolved_deps) => Ok((dep.as_str(), resolved_deps)),
                Err(e) => Err(e.into()),
            }
        }))
    }

    /// Get the general + backend-specific Python dependencies for the given backend.
    pub fn all_python_depends(&self, backend: Backend) -> Result<Vec<String>> {
        self.general_python_depends()
            .map(|deps| Ok(deps?.0.to_owned()))
            .chain(
                self.backend_python_depends(backend)
                    .map(|deps| Ok(deps?.0.to_owned())),
            )
            .collect::<Result<Vec<_>>>()
    }

    /// Get the general + backend-specific kernel dependencies for the given backend.
    pub fn all_kernel_depends(&self, backend: Backend) -> Vec<KernelDependency> {
        let general = self.kernel_depends.iter().flatten().cloned();
        let backend_depends = match backend {
            Backend::Cuda => self.cuda.as_ref().and_then(|c| c.kernel_depends.as_ref()),
            Backend::Neuron => self.neuron.as_ref().and_then(|n| n.kernel_depends.as_ref()),
            Backend::Xpu => self.xpu.as_ref().and_then(|x| x.kernel_depends.as_ref()),
            _ => None,
        };
        general
            .chain(backend_depends.into_iter().flatten().cloned())
            .collect()
    }
}

pub struct CudaGeneral {
    pub minver: Option<Version<2>>,
    pub maxver: Option<Version<2>>,
    pub kernel_depends: Option<Vec<KernelDependency>>,
    pub python_depends: Option<Vec<String>>,
}

pub struct XpuGeneral {
    pub kernel_depends: Option<Vec<KernelDependency>>,
    pub python_depends: Option<Vec<String>>,
}

pub struct NeuronGeneral {
    pub kernel_depends: Option<Vec<KernelDependency>>,
    pub python_depends: Option<Vec<String>>,
}

pub struct TpuGeneral {
    pub python_depends: Option<Vec<String>>,
}

pub struct Hub {
    pub repo_id: Option<String>,
    pub branch: Option<String>,
}

pub struct Torch {
    pub include: Option<Vec<String>>,
    pub minver: Option<Version<2>>,
    pub maxver: Option<Version<2>>,
    pub pyext: Option<Vec<String>>,
    pub src: Vec<PathBuf>,
    pub stable_abi: Option<TorchAbi>,
    pub cxx_flags: Option<Vec<String>>,
}

/// Torch stable ABI version: a single version for all backends, or per-backend
/// versions. Backends absent from a mapping are built without the stable ABI.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum TorchAbi {
    All(Version<2>),
    Mapping(BTreeMap<Backend, Version<2>>),
}

impl TorchAbi {
    /// Stable ABI version to target for `backend`, or `None` if it should be
    /// built without the stable ABI.
    pub fn for_backend(&self, backend: Backend) -> Option<&Version<2>> {
        match self {
            TorchAbi::All(version) => Some(version),
            TorchAbi::Mapping(mapping) => mapping.get(&backend),
        }
    }
}

fn data_extensions(py_ext: Option<&[String]>) -> Option<Vec<String>> {
    match py_ext {
        Some(exts) => {
            let extensions = exts
                .iter()
                .filter(|&ext| ext != "py" && ext != "pyi")
                .cloned()
                .collect_vec();
            if extensions.is_empty() {
                None
            } else {
                Some(extensions)
            }
        }

        None => None,
    }
}

impl Torch {
    pub fn data_extensions(&self) -> Option<Vec<String>> {
        data_extensions(self.pyext.as_deref())
    }
}

pub struct TorchNoarch {
    pub pyext: Option<Vec<String>>,
    /// CUDA capabilities to write into metadata.
    pub cuda_capabilities: Option<Vec<String>>,

    /// ROCM archs to write into metadata.
    pub rocm_archs: Option<Vec<String>>,
}

impl TorchNoarch {
    pub fn data_extensions(&self) -> Option<Vec<String>> {
        data_extensions(self.pyext.as_deref())
    }
}

pub struct TvmFfi {
    pub include: Option<Vec<String>>,
    pub pyext: Option<Vec<String>>,
    pub src: Vec<PathBuf>,
    pub cxx_flags: Option<Vec<String>>,
}

impl TvmFfi {
    pub fn data_extensions(&self) -> Option<Vec<String>> {
        data_extensions(self.pyext.as_deref())
    }
}

pub enum Kernel {
    Cpu {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    Cuda {
        cuda_capabilities: Option<Vec<String>>,
        cuda_flags: Option<Vec<String>>,
        cuda_minver: Option<Version<2>>,
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    Metal {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    Rocm {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        rocm_archs: Option<Vec<String>>,
        hip_flags: Option<Vec<String>>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    Xpu {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        sycl_flags: Option<Vec<String>>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
}

impl Kernel {
    pub fn cxx_flags(&self) -> Option<&[String]> {
        match self {
            Kernel::Cpu { cxx_flags, .. }
            | Kernel::Cuda { cxx_flags, .. }
            | Kernel::Metal { cxx_flags, .. }
            | Kernel::Rocm { cxx_flags, .. }
            | Kernel::Xpu { cxx_flags, .. } => cxx_flags.as_deref(),
        }
    }

    pub fn include(&self) -> Option<&[String]> {
        match self {
            Kernel::Cpu { include, .. }
            | Kernel::Cuda { include, .. }
            | Kernel::Metal { include, .. }
            | Kernel::Rocm { include, .. }
            | Kernel::Xpu { include, .. } => include.as_deref(),
        }
    }

    pub fn sycl_flags(&self) -> Option<&[String]> {
        match self {
            Kernel::Xpu { sycl_flags, .. } => sycl_flags.as_deref(),
            _ => None,
        }
    }

    pub fn backend(&self) -> Backend {
        match self {
            Kernel::Cpu { .. } => Backend::Cpu,
            Kernel::Cuda { .. } => Backend::Cuda,
            Kernel::Metal { .. } => Backend::Metal,
            Kernel::Rocm { .. } => Backend::Rocm,
            Kernel::Xpu { .. } => Backend::Xpu,
        }
    }

    pub fn depends(&self) -> &[Dependency] {
        match self {
            Kernel::Cpu { depends, .. }
            | Kernel::Cuda { depends, .. }
            | Kernel::Metal { depends, .. }
            | Kernel::Rocm { depends, .. }
            | Kernel::Xpu { depends, .. } => depends,
        }
    }

    pub fn src(&self) -> &[String] {
        match self {
            Kernel::Cpu { src, .. }
            | Kernel::Cuda { src, .. }
            | Kernel::Metal { src, .. }
            | Kernel::Rocm { src, .. }
            | Kernel::Xpu { src, .. } => src,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub enum Backend {
    Cann,
    Cpu,
    Cuda,
    Metal,
    Neuron,
    Rocm,
    Tpu,
    Xpu,
}

impl Backend {
    pub const fn all() -> [Backend; 8] {
        [
            Backend::Cann,
            Backend::Cpu,
            Backend::Cuda,
            Backend::Metal,
            Backend::Neuron,
            Backend::Rocm,
            Backend::Tpu,
            Backend::Xpu,
        ]
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Backend::Cann => "cann",
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Neuron => "neuron",
            Backend::Rocm => "rocm",
            Backend::Tpu => "tpu",
            Backend::Xpu => "xpu",
        }
    }
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cann => write!(f, "cann"),
            Backend::Cpu => write!(f, "cpu"),
            Backend::Cuda => write!(f, "cuda"),
            Backend::Metal => write!(f, "metal"),
            Backend::Neuron => write!(f, "neuron"),
            Backend::Rocm => write!(f, "rocm"),
            Backend::Tpu => write!(f, "tpu"),
            Backend::Xpu => write!(f, "xpu"),
        }
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cann" => Ok(Backend::Cann),
            "cpu" => Ok(Backend::Cpu),
            "cuda" => Ok(Backend::Cuda),
            "metal" => Ok(Backend::Metal),
            "neuron" => Ok(Backend::Neuron),
            "rocm" => Ok(Backend::Rocm),
            "tpu" => Ok(Backend::Tpu),
            "xpu" => Ok(Backend::Xpu),
            _ => Err(format!("Unknown backend: {s}")),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Cannot migrate configuration: {reason:?}")]
    Migration { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn general_with_cuda_deps() -> General {
        General {
            name: KernelName::new("test-kernel").unwrap(),
            version: 1,
            license: "apache-2.0".to_string(),
            upstream: None,
            source: None,
            backends: vec![Backend::Tpu],
            hub: None,
            kernel_depends: None,
            python_depends: None,
            cuda: Some(CudaGeneral {
                minver: None,
                maxver: None,
                kernel_depends: None,
                python_depends: Some(vec!["nvidia-cutlass-dsl".to_string()]),
            }),
            neuron: None,
            tpu: None,
            xpu: None,
        }
    }

    #[test]
    fn backend_python_depends_resolves_cuda() {
        let general = general_with_cuda_deps();
        let deps = general
            .backend_python_depends(Backend::Cuda)
            .map(|dep| dep.map(|(name, _)| name.to_string()))
            .collect::<Result<Vec<_>>>()
            .unwrap();

        assert_eq!(deps, vec!["nvidia-cutlass-dsl".to_string()]);
    }

    #[test]
    fn backend_python_depends_empty_for_backend_without_deps() {
        let general = general_with_cuda_deps();

        assert!(
            general
                .backend_python_depends(Backend::Cpu)
                .next()
                .is_none()
        );
    }

    /// The minimum `kernels` version is derived from the features that a
    /// kernel uses, so kernel authors must not be able to set it themselves.
    #[test]
    fn general_minver_is_rejected() {
        let toml = r#"
            [general]
            name = "test-kernel"
            version = 1
            edition = 5
            license = "apache-2.0"
            backends = ["cuda"]
            minver = "0.11.0"

            [torch]
            src = []
        "#;

        let err = toml::from_str::<v5::Build>(toml).unwrap_err().to_string();
        assert!(err.contains("unknown field `minver`"), "{err}");
    }
}
