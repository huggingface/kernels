use std::{collections::HashMap, sync::LazyLock};

use eyre::Result;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::Backend;

pub static PYTHON_DEPENDENCIES: LazyLock<PythonDependencies> =
    LazyLock::new(|| serde_json::from_str(include_str!("../python_dependencies.json")).unwrap());

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[non_exhaustive]
#[serde(rename_all = "lowercase")]
pub enum Dependency {
    #[serde(rename = "cutlass_2_10")]
    Cutlass2_10,
    #[serde(rename = "cutlass_3_5")]
    Cutlass3_5,
    #[serde(rename = "cutlass_3_6")]
    Cutlass3_6,
    #[serde(rename = "cutlass_3_8")]
    Cutlass3_8,
    #[serde(rename = "cutlass_3_9")]
    Cutlass3_9,
    #[serde(rename = "cutlass_4_0")]
    Cutlass4_0,
    #[serde(rename = "sycl_tla")]
    SyclTla,
    #[serde(rename = "metal-cpp")]
    MetalCpp,
    Torch,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PythonDependencies {
    general: HashMap<String, PythonDependency>,
    backends: HashMap<Backend, HashMap<String, PythonDependency>>,
}

impl PythonDependencies {
    pub fn get_dependency(&self, dependency: &str) -> Result<&[String], DependencyError> {
        match self.general.get(dependency) {
            None => Err(DependencyError::GeneralDependency {
                dependency: dependency.to_string(),
            }),
            Some(dep) => Ok(&dep.python),
        }
    }

    pub fn get_backend_dependency(
        &self,
        backend: Backend,
        dependency: &str,
    ) -> Result<&[String], DependencyError> {
        let backend_deps = match self.backends.get(&backend) {
            None => {
                return Err(DependencyError::Backend {
                    backend: backend.to_string(),
                })
            }
            Some(backend_deps) => backend_deps,
        };
        match backend_deps.get(dependency) {
            None => Err(DependencyError::Dependency {
                backend: backend.to_string(),
                dependency: dependency.to_string(),
            }),
            Some(dep) => Ok(&dep.python),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct PythonDependency {
    nix: Vec<String>,
    python: Vec<String>,
}

#[derive(Debug, Error)]
pub enum DependencyError {
    #[error("No dependencies are defined for backend: {backend:?}")]
    Backend { backend: String },
    #[error("Unknown dependency `{dependency:?}` for backend `{backend:?}`")]
    Dependency { backend: String, dependency: String },
    #[error("Unknown dependency: `{dependency:?}`")]
    GeneralDependency { dependency: String },
}
