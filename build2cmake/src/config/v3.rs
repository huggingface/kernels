use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{Dependency, KernelName};
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

    pub version: Option<usize>,

    pub license: Option<String>,

    pub backends: Vec<Backend>,

    pub cuda: Option<CudaGeneral>,

    pub hub: Option<Hub>,

    pub python_depends: Option<Vec<String>>,

    pub xpu: Option<XpuGeneral>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct CudaGeneral {
    pub minver: Option<Version>,
    pub maxver: Option<Version>,
    pub python_depends: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct XpuGeneral {
    pub python_depends: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Hub {
    pub repo_id: Option<String>,
    pub branch: Option<String>,
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Rocm,
    Xpu,
}

impl From<Build> for super::Build {
    fn from(build: Build) -> Self {
        let kernels: HashMap<String, super::Kernel> = build
            .kernels
            .into_iter()
            .map(|(k, v)| (k, v.into()))
            .collect();

        Self {
            general: build.general.into(),
            torch: build.torch.map(Into::into),
            kernels,
        }
    }
}

impl From<General> for super::General {
    fn from(general: General) -> Self {
        Self {
            name: general.name,
            version: general.version,
            license: general.license,
            backends: general.backends.into_iter().map(Into::into).collect(),
            cuda: general.cuda.map(Into::into),
            hub: general.hub.map(Into::into),
            python_depends: general.python_depends,
            xpu: general.xpu.map(Into::into),
        }
    }
}

impl From<CudaGeneral> for super::CudaGeneral {
    fn from(cuda: CudaGeneral) -> Self {
        Self {
            minver: cuda.minver,
            maxver: cuda.maxver,
            python_depends: cuda.python_depends,
        }
    }
}

impl From<XpuGeneral> for super::XpuGeneral {
    fn from(xpu: XpuGeneral) -> Self {
        Self {
            python_depends: xpu.python_depends,
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

impl From<Backend> for super::Backend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Cpu => super::Backend::Cpu,
            Backend::Cuda => super::Backend::Cuda,
            Backend::Metal => super::Backend::Metal,
            Backend::Rocm => super::Backend::Rocm,
            Backend::Xpu => super::Backend::Xpu,
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

impl From<super::Build> for Build {
    fn from(build: super::Build) -> Self {
        Self {
            general: build.general.into(),
            torch: build.torch.map(Into::into),
            kernels: build
                .kernels
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

impl From<super::General> for General {
    fn from(general: super::General) -> Self {
        Self {
            name: general.name,
            version: general.version,
            license: general.license,
            backends: general.backends.into_iter().map(Into::into).collect(),
            cuda: general.cuda.map(Into::into),
            hub: general.hub.map(Into::into),
            python_depends: general.python_depends,
            xpu: general.xpu.map(Into::into),
        }
    }
}

impl From<super::CudaGeneral> for CudaGeneral {
    fn from(cuda: super::CudaGeneral) -> Self {
        Self {
            minver: cuda.minver,
            maxver: cuda.maxver,
            python_depends: cuda.python_depends,
        }
    }
}

impl From<super::XpuGeneral> for XpuGeneral {
    fn from(xpu: super::XpuGeneral) -> Self {
        Self {
            python_depends: xpu.python_depends,
        }
    }
}

impl From<super::Hub> for Hub {
    fn from(hub: super::Hub) -> Self {
        Self {
            repo_id: hub.repo_id,
            branch: hub.branch,
        }
    }
}

impl From<super::Torch> for Torch {
    fn from(torch: super::Torch) -> Self {
        Self {
            include: torch.include,
            minver: torch.minver,
            maxver: torch.maxver,
            pyext: torch.pyext,
            src: torch.src,
        }
    }
}

impl From<super::Backend> for Backend {
    fn from(backend: super::Backend) -> Self {
        match backend {
            super::Backend::Cpu => Backend::Cpu,
            super::Backend::Cuda => Backend::Cuda,
            super::Backend::Metal => Backend::Metal,
            super::Backend::Rocm => Backend::Rocm,
            super::Backend::Xpu => Backend::Xpu,
        }
    }
}

impl From<super::Kernel> for Kernel {
    fn from(kernel: super::Kernel) -> Self {
        match kernel {
            super::Kernel::Cpu {
                cxx_flags,
                depends,
                include,
                src,
            } => Kernel::Cpu {
                cxx_flags,
                depends,
                include,
                src,
            },
            super::Kernel::Cuda {
                cuda_capabilities,
                cuda_flags,
                cuda_minver,
                cxx_flags,
                depends,
                include,
                src,
            } => Kernel::Cuda {
                cuda_capabilities,
                cuda_flags,
                cuda_minver,
                cxx_flags,
                depends,
                include,
                src,
            },
            super::Kernel::Metal {
                cxx_flags,
                depends,
                include,
                src,
            } => Kernel::Metal {
                cxx_flags,
                depends,
                include,
                src,
            },
            super::Kernel::Rocm {
                cxx_flags,
                depends,
                rocm_archs,
                hip_flags,
                include,
                src,
            } => Kernel::Rocm {
                cxx_flags,
                depends,
                rocm_archs,
                hip_flags,
                include,
                src,
            },
            super::Kernel::Xpu {
                cxx_flags,
                depends,
                sycl_flags,
                include,
                src,
            } => Kernel::Xpu {
                cxx_flags,
                depends,
                sycl_flags,
                include,
                src,
            },
        }
    }
}
