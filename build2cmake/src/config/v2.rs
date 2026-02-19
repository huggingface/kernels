use std::{
    collections::{BTreeSet, HashMap},
    fmt::Display,
    path::PathBuf,
};

use eyre::Result;
use serde::{Deserialize, Serialize};

use super::{Backend, Dependency, KernelName};
use crate::version::Version;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub general: General,
    pub torch: Option<Torch>,

    #[serde(rename = "kernel", default)]
    pub kernels: HashMap<String, Kernel>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct General {
    pub name: KernelName,
    #[serde(default)]
    pub universal: bool,

    pub cuda_maxver: Option<Version>,

    pub cuda_minver: Option<Version>,

    pub hub: Option<Hub>,

    pub python_depends: Option<Vec<PythonDependency>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Hub {
    pub repo_id: Option<String>,
    pub branch: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub enum PythonDependency {
    Einops,
    NvidiaCutlassDsl,
}

impl Display for PythonDependency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PythonDependency::Einops => write!(f, "einops"),
            PythonDependency::NvidiaCutlassDsl => write!(f, "nvidia-cutlass-dsl"),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Torch {
    pub include: Option<Vec<String>>,
    pub minver: Option<Version>,
    pub maxver: Option<Version>,
    pub pyext: Option<Vec<String>>,

    #[serde(default)]
    pub src: Vec<PathBuf>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case", tag = "backend")]
pub enum Kernel {
    #[serde(rename_all = "kebab-case")]
    Cpu {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    #[serde(rename_all = "kebab-case")]
    Cuda {
        cuda_capabilities: Option<Vec<String>>,
        cuda_flags: Option<Vec<String>>,
        cuda_minver: Option<Version>,
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    #[serde(rename_all = "kebab-case")]
    Metal {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    #[serde(rename_all = "kebab-case")]
    Rocm {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        rocm_archs: Option<Vec<String>>,
        hip_flags: Option<Vec<String>>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
    #[serde(rename_all = "kebab-case")]
    Xpu {
        cxx_flags: Option<Vec<String>>,
        depends: Vec<Dependency>,
        sycl_flags: Option<Vec<String>>,
        include: Option<Vec<String>>,
        src: Vec<String>,
    },
}

impl TryFrom<Build> for super::Build {
    type Error = eyre::Error;

    fn try_from(build: Build) -> Result<Self> {
        let kernels: HashMap<String, super::Kernel> = build
            .kernels
            .into_iter()
            .map(|(k, v)| (k, v.into()))
            .collect();

        let backends = if build.general.universal {
            vec![
                Backend::Cpu,
                Backend::Cuda,
                Backend::Metal,
                Backend::Neuron,
                Backend::Rocm,
                Backend::Xpu,
            ]
        } else {
            let backend_set: BTreeSet<Backend> =
                kernels.values().map(|kernel| kernel.backend()).collect();
            backend_set.into_iter().collect()
        };

        Ok(Self {
            general: General::from_v2(build.general, backends),
            torch: build.torch.map(Into::into),
            kernels,
        })
    }
}

impl General {
    fn from_v2(general: General, backends: Vec<Backend>) -> super::General {
        let cuda = if general.cuda_minver.is_some() || general.cuda_maxver.is_some() {
            Some(super::CudaGeneral {
                minver: general.cuda_minver,
                maxver: general.cuda_maxver,
                python_depends: None,
            })
        } else {
            None
        };

        super::General {
            name: general.name,
            version: None,
            license: None,
            backends,
            cuda,
            hub: general.hub.map(Into::into),
            neuron: None,
            python_depends: None,
            xpu: None,
        }
    }
}

impl From<Hub> for super::Hub {
    fn from(hub: Hub) -> Self {
        Self {
            repo_id: hub.repo_id,
            branch: hub.branch,
        }
    }
}

impl From<Torch> for super::Torch {
    fn from(torch: Torch) -> Self {
        Self {
            include: torch.include,
            minver: torch.minver,
            maxver: torch.maxver,
            pyext: torch.pyext,
            src: torch.src,
        }
    }
}

impl From<Kernel> for super::Kernel {
    fn from(kernel: Kernel) -> Self {
        match kernel {
            Kernel::Cpu {
                cxx_flags,
                depends,
                include,
                src,
            } => super::Kernel::Cpu {
                cxx_flags,
                depends,
                include,
                src,
            },
            Kernel::Cuda {
                cuda_capabilities,
                cuda_flags,
                cuda_minver,
                cxx_flags,
                depends,
                include,
                src,
            } => super::Kernel::Cuda {
                cuda_capabilities,
                cuda_flags,
                cuda_minver,
                cxx_flags,
                depends,
                include,
                src,
            },
            Kernel::Metal {
                cxx_flags,
                depends,
                include,
                src,
            } => super::Kernel::Metal {
                cxx_flags,
                depends,
                include,
                src,
            },
            Kernel::Rocm {
                cxx_flags,
                depends,
                rocm_archs,
                hip_flags,
                include,
                src,
            } => super::Kernel::Rocm {
                cxx_flags,
                depends,
                rocm_archs,
                hip_flags,
                include,
                src,
            },
            Kernel::Xpu {
                cxx_flags,
                depends,
                sycl_flags,
                include,
                src,
            } => super::Kernel::Xpu {
                cxx_flags,
                depends,
                sycl_flags,
                include,
                src,
            },
        }
    }
}
