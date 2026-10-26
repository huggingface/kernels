use std::str::FromStr;

use kernels_common::git::Oid;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A git object identifier.
#[pyclass(name = "Oid", frozen, eq, hash, ord)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct PyOid {
    inner: Oid,
}

impl From<Oid> for PyOid {
    fn from(inner: Oid) -> Self {
        Self { inner }
    }
}

impl PyOid {
    pub(crate) fn into_inner(self) -> Oid {
        self.inner
    }
}

/// Parse a git object id, mapping a parse failure to a Python `ValueError`.
pub(crate) fn parse_oid(s: &str) -> PyResult<Oid> {
    Oid::from_str(s).map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pymethods]
impl PyOid {
    /// Parse a full SHA-1 or SHA-256 object id.
    ///
    /// Abbreviated identifiers are rejected: an object id must identify the
    /// object unambiguously and permanently.
    #[staticmethod]
    #[pyo3(name = "from_str")]
    fn py_from_str(s: &str) -> PyResult<Self> {
        parse_oid(s).map(Into::into)
    }

    fn __str__(&self) -> &str {
        self.inner.as_str()
    }

    fn __repr__(&self) -> String {
        format!("Oid({:?})", self.inner.as_str())
    }
}
