use std::path::PathBuf;

use kernels_data::config::{Build, General};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{PyBackend, PyKernelDependency};

/// General kernel configuration common to all backends.
#[pyclass(name = "General", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyGeneral {
    backends: Vec<PyBackend>,
}

impl From<&General> for PyGeneral {
    fn from(general: &General) -> Self {
        Self {
            backends: general.backends.iter().copied().map(Into::into).collect(),
        }
    }
}

#[pymethods]
impl PyGeneral {
    #[getter]
    fn backends(&self) -> Vec<PyBackend> {
        self.backends.clone()
    }

    fn __repr__(&self) -> String {
        format!("General(backends={:?})", self.backends)
    }
}

/// Parsed and validated `build.toml` configuration for a kernel.
#[pyclass(name = "Build", frozen)]
pub(crate) struct PyBuild {
    inner: Build,
}

#[pymethods]
impl PyBuild {
    /// Parse and validate the `build.toml` in `kernel_dir`.
    ///
    /// Raises `ValueError` if the build configuration cannot be parsed or
    /// validated.
    #[staticmethod]
    fn open(kernel_dir: PathBuf) -> PyResult<PyBuild> {
        Build::open(&kernel_dir)
            .map(|inner| PyBuild { inner })
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "Cannot parse build configuration in `{}`: {err:#}",
                    kernel_dir.display()
                ))
            })
    }

    #[getter]
    fn general(&self) -> PyGeneral {
        PyGeneral::from(&self.inner.general)
    }

    /// Get the general + backend-specific kernel dependencies for `backend`.
    fn all_kernel_depends(&self, backend: PyBackend) -> Vec<PyKernelDependency> {
        self.inner
            .general
            .all_kernel_depends(backend.into())
            .into_iter()
            .map(Into::into)
            .collect()
    }
}
