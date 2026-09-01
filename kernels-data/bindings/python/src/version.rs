use kernels_data::version::Version;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A dotted numeric version (e.g. `12.8.0`).
#[pyclass(name = "Version", frozen, eq, ord, hash)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct PyVersion {
    // We are storing this as a `Box<[usize]>` to allow storing kernels-data
    // `Version` in directly in `PyVersion` as well.
    inner: Box<[usize]>,
}

impl<const N: usize> From<Version<N>> for PyVersion {
    fn from(version: Version<N>) -> Self {
        Self {
            inner: Box::from(&*version),
        }
    }
}

#[pymethods]
impl PyVersion {
    /// Parse a version string with exactly `n_components` dotted components
    /// (e.g. `12.8.0` for `n_components=3`).
    #[staticmethod]
    #[pyo3(name = "from_str")]
    fn py_from_str(s: &str, n_components: usize) -> PyResult<Self> {
        let version = s.trim();
        if version.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Cannot parse version `{s}`: empty version string"
            )));
        }
        let mut parts = Vec::new();
        for part in version.split('.') {
            let component: usize = part.parse().map_err(|_| {
                PyValueError::new_err(format!(
                    "Cannot parse version `{s}`: version must consist of numbers"
                ))
            })?;
            parts.push(component);
        }
        if parts.len() != n_components {
            return Err(PyValueError::new_err(format!(
                "Version `{s}` has {} components, expected {n_components}",
                parts.len()
            )));
        }
        Ok(PyVersion {
            inner: parts.into_boxed_slice(),
        })
    }

    fn __str__(&self) -> String {
        self.inner
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(".")
    }

    fn __repr__(&self) -> String {
        format!("Version('{}')", self.__str__())
    }
}
