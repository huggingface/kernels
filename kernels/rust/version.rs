use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use kernels_data::version::Version;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A dotted numeric version (e.g. `12.8.0`). Versions that only differ in
/// trailing zeros compare as equal.
#[pyclass(name = "Version", frozen, eq, ord, hash)]
#[derive(Clone, Debug)]
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

// A parsed version string does not carry `Version<N>`'s guarantee that both
// sides of a comparison have the same number of components, so compare
// component-wise with missing components counting as zero (`0.17` and
// `0.17.0` are the same version).
impl Ord for PyVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        let n = self.inner.len().max(other.inner.len());
        for i in 0..n {
            let lhs = self.inner.get(i).copied().unwrap_or(0);
            let rhs = other.inner.get(i).copied().unwrap_or(0);
            match lhs.cmp(&rhs) {
                Ordering::Equal => (),
                ordering => return ordering,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for PyVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for PyVersion {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PyVersion {}

impl Hash for PyVersion {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Skip trailing zeros so that versions that compare as equal hash
        // identically.
        let trailing_zeros = self.inner.iter().rev().take_while(|&&x| x == 0).count();
        self.inner[..self.inner.len() - trailing_zeros].hash(state);
    }
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
