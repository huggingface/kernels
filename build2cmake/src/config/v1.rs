use std::{
    collections::{BTreeSet, HashMap},
    fmt::Display,
    path::PathBuf,
};

use eyre::{bail, Result};
use serde::Deserialize;

use super::{Backend, Dependency, KernelName};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub general: General,
    pub torch: Option<Torch>,

    #[serde(rename = "kernel", default)]
    pub kernels: HashMap<String, Kernel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct General {
    pub name: KernelName,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Torch {
    pub include: Option<Vec<String>>,
    pub pyext: Option<Vec<String>>,

    #[serde(default)]
    pub src: Vec<PathBuf>,

    #[serde(default)]
    pub universal: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Kernel {
    pub cuda_capabilities: Option<Vec<String>>,
    pub rocm_archs: Option<Vec<String>>,
    #[serde(default)]
    pub language: Language,
    pub depends: Vec<Dependency>,
    pub include: Option<Vec<String>>,
    pub src: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub enum Language {
    #[default]
    Cuda,
    CudaHipify,
    Metal,
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Cuda => f.write_str("cuda"),
            Language::CudaHipify => f.write_str("cuda-hipify"),
            Language::Metal => f.write_str("metal"),
        }
    }
}

impl TryFrom<Build> for super::Build {
    type Error = eyre::Error;

    fn try_from(build: Build) -> Result<Self> {
        let universal = build
            .torch
            .as_ref()
            .map(|torch| torch.universal)
            .unwrap_or(false);

        let kernels = convert_kernels(build.kernels)?;

        let backends = if universal {
            vec![
                Backend::Cpu,
                Backend::Cuda,
                Backend::Metal,
                Backend::Rocm,
                Backend::Xpu,
            ]
        } else {
            let backend_set: BTreeSet<Backend> =
                kernels.values().map(|kernel| kernel.backend()).collect();
            backend_set.into_iter().collect()
        };

        Ok(Self {
            general: super::General {
                name: build.general.name,
                version: None,
                license: None,
                backends,
                hub: None,
                python_depends: None,
                cuda: None,
                xpu: None,
            },
            torch: build.torch.map(Into::into),
            kernels,
        })
    }
}

fn convert_kernels(v1_kernels: HashMap<String, Kernel>) -> Result<HashMap<String, super::Kernel>> {
    let mut kernels = HashMap::new();

    for (name, kernel) in v1_kernels {
        if kernel.language == Language::CudaHipify {
            // We need to add an affix to avoid conflict with the CUDA kernel.
            let rocm_name = format!("{name}_rocm");
            if kernels.contains_key(&rocm_name) {
                bail!("Found an existing kernel with name `{rocm_name}` while expanding `{name}`")
            }

            kernels.insert(
                format!("{name}_rocm"),
                super::Kernel::Rocm {
                    cxx_flags: None,
                    rocm_archs: kernel.rocm_archs,
                    hip_flags: None,
                    depends: kernel.depends.clone(),
                    include: kernel.include.clone(),
                    src: kernel.src.clone(),
                },
            );
        }

        kernels.insert(
            name,
            super::Kernel::Cuda {
                cuda_capabilities: kernel.cuda_capabilities,
                cuda_flags: None,
                cuda_minver: None,
                cxx_flags: None,
                depends: kernel.depends,
                include: kernel.include,
                src: kernel.src,
            },
        );
    }

    Ok(kernels)
}

impl From<Torch> for super::Torch {
    fn from(torch: Torch) -> Self {
        Self {
            include: torch.include,
            minver: None,
            maxver: None,
            pyext: torch.pyext,
            src: torch.src,
            bundle_dep_includes: None,
        }
    }
}
