use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A dotted numeric version (e.g. `12.8.0`). Trailing zeros are stripped
/// during normalization.
#[pyclass(name = "Version", frozen, eq, ord, hash)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct PyVersion {
    // We are storing this as a `Box<[usize]>` to allow storing kernels-data
    // `Version` in directly in `PyVersion` as well.
    inner: Box<[usize]>,
}

#[pymethods]
impl PyVersion {
    /// Parse a version string of the form `X`, `X.Y`, `X.Y.Z`, ...
    #[staticmethod]
    #[pyo3(name = "from_str")]
    fn py_from_str(s: &str) -> PyResult<Self> {
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
        // Remove trailing zeros for normalization.
        let trailing_zeros = parts.iter().rev().take_while(|&&x| x == 0).count();
        parts.truncate(parts.len() - trailing_zeros);
        Ok(Self {
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
