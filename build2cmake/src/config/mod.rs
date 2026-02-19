use std::{collections::HashMap, fmt::Display, path::PathBuf, str::FromStr};

use eyre::Result;
use serde::{Deserialize, Serialize};

mod deps;
pub use deps::Dependency;

mod compat;
pub use compat::BuildCompat;

mod name;
pub use name::KernelName;

mod v1;
mod v2;
pub(crate) mod v3;

use itertools::Itertools;

use crate::version::Version;

pub struct Build {
    pub general: General,
    pub kernels: HashMap<String, Kernel>,
    pub torch: Option<Torch>,
}

impl Build {
    pub fn is_noarch(&self) -> bool {
        self.kernels.is_empty()
    }
}

pub struct General {
    pub name: KernelName,
    pub version: Option<usize>,

    /// Hugging Face Hub license identifier.
    pub license: Option<String>,

    pub backends: Vec<Backend>,
    pub hub: Option<Hub>,
    pub python_depends: Option<Vec<String>>,

    pub cuda: Option<CudaGeneral>,
    pub neuron: Option<NeuronGeneral>,
    pub xpu: Option<XpuGeneral>,
}

impl General {
    pub fn python_depends(&self) -> Box<dyn Iterator<Item = Result<String>> + '_> {
        let general_python_deps = match self.python_depends.as_ref() {
            Some(deps) => deps,
            None => {
                return Box::new(std::iter::empty());
            }
        };

        Box::new(general_python_deps.iter().flat_map(move |dep| {
            match deps::PYTHON_DEPENDENCIES.get_dependency(dep) {
                Ok(deps) => deps.iter().map(|s| Ok(s.clone())).collect::<Vec<_>>(),
                Err(e) => vec![Err(e.into())],
            }
        }))
    }

    pub fn backend_python_depends(
        &self,
        backend: Backend,
    ) -> Box<dyn Iterator<Item = Result<String>> + '_> {
        let backend_python_deps = match backend {
            Backend::Cuda => self
                .cuda
                .as_ref()
                .and_then(|cuda| cuda.python_depends.as_ref()),
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

        Box::new(backend_python_deps.iter().flat_map(move |dep| {
            match deps::PYTHON_DEPENDENCIES.get_backend_dependency(backend, dep) {
                Ok(deps) => deps.iter().map(|s| Ok(s.clone())).collect::<Vec<_>>(),
                Err(e) => vec![Err(e.into())],
            }
        }))
    }
}

pub struct CudaGeneral {
    pub minver: Option<Version>,
    pub maxver: Option<Version>,
    pub python_depends: Option<Vec<String>>,
}

pub struct XpuGeneral {
    pub python_depends: Option<Vec<String>>,
}

pub struct NeuronGeneral {
    pub python_depends: Option<Vec<String>>,
}

pub struct Hub {
    pub repo_id: Option<String>,
    pub branch: Option<String>,
}

pub struct Torch {
    pub include: Option<Vec<String>>,
    pub minver: Option<Version>,
    pub maxver: Option<Version>,
    pub pyext: Option<Vec<String>>,
    pub src: Vec<PathBuf>,
}

impl Torch {
    pub fn data_extensions(&self) -> Option<Vec<String>> {
        match self.pyext.as_ref() {
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
        cuda_minver: Option<Version>,
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
    Cpu,
    Cuda,
    Metal,
    Neuron,
    Rocm,
    Xpu,
}

impl Backend {
    pub const fn all() -> [Backend; 6] {
        [
            Backend::Cpu,
            Backend::Cuda,
            Backend::Metal,
            Backend::Neuron,
            Backend::Rocm,
            Backend::Xpu,
        ]
    }
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::Cpu => write!(f, "cpu"),
            Backend::Cuda => write!(f, "cuda"),
            Backend::Metal => write!(f, "metal"),
            Backend::Neuron => write!(f, "neuron"),
            Backend::Rocm => write!(f, "rocm"),
            Backend::Xpu => write!(f, "xpu"),
        }
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Backend::Cpu),
            "cuda" => Ok(Backend::Cuda),
            "metal" => Ok(Backend::Metal),
            "neuron" => Ok(Backend::Neuron),
            "rocm" => Ok(Backend::Rocm),
            "xpu" => Ok(Backend::Xpu),
            _ => Err(format!("Unknown backend: {s}")),
        }
    }
}
