use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use kernels_data::git::Oid;
use kernels_data::lock::{KernelLock, KernelLocks, KernelPaths, NixKernelLock, NixKernelLocks};
use pyo3::Bound as PyBound;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

use crate::PyKernelDependency;

/// Parse a git object id, mapping a parse failure to a Python `ValueError`.
fn parse_oid(s: &str) -> PyResult<Oid> {
    Oid::from_str(s).map_err(|err| PyValueError::new_err(err.to_string()))
}

/// A locked kernel revision.
#[pyclass(name = "KernelLock", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyKernelLock {
    commit: Oid,
}

impl From<KernelLock> for PyKernelLock {
    fn from(lock: KernelLock) -> Self {
        Self {
            commit: lock.commit,
        }
    }
}

impl From<PyKernelLock> for KernelLock {
    fn from(lock: PyKernelLock) -> Self {
        Self {
            commit: lock.commit,
        }
    }
}

#[pymethods]
impl PyKernelLock {
    #[new]
    fn new(commit: &str) -> PyResult<Self> {
        Ok(Self {
            commit: parse_oid(commit)?,
        })
    }

    #[getter]
    fn commit(&self) -> &str {
        self.commit.as_str()
    }

    /// Parse a `KernelLock` from a JSON string.
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(s: &str) -> PyResult<Self> {
        let lock: KernelLock = serde_json::from_str(s)
            .map_err(|err| PyValueError::new_err(format!("Cannot parse KernelLock: {err:#}")))?;
        Ok(lock.into())
    }

    /// Serialize the lock to a pretty-printed JSON string.
    fn to_json(&self) -> PyResult<String> {
        let lock: KernelLock = self.clone().into();
        serde_json::to_string_pretty(&lock)
            .map_err(|err| PyValueError::new_err(format!("Cannot serialize KernelLock: {err:#}")))
    }

    fn __repr__(&self) -> String {
        format!("KernelLock(commit={:?})", self.commit.as_str())
    }
}

/// Multiple kernel locks keyed by the dependency they resolve.
#[pyclass(name = "KernelLocks", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyKernelLocks {
    locks: BTreeMap<PyKernelDependency, PyKernelLock>,
}

/// Iterator over the dependencies of a [`PyKernelLocks`] or [`PyKernelPaths`].
#[pyclass(name = "KernelDependencyIterator")]
struct PyKernelDependencyIterator {
    dependencies: std::vec::IntoIter<PyKernelDependency>,
}

#[pymethods]
impl PyKernelDependencyIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyKernelDependency> {
        self.dependencies.next()
    }
}

impl From<KernelLocks> for PyKernelLocks {
    fn from(locks: KernelLocks) -> Self {
        Self {
            locks: locks
                .locks
                .into_iter()
                .map(|(dep, lock)| (dep.into(), lock.into()))
                .collect(),
        }
    }
}

impl From<PyKernelLocks> for KernelLocks {
    fn from(locks: PyKernelLocks) -> Self {
        Self {
            locks: locks
                .locks
                .into_iter()
                .map(|(dep, lock)| (dep.into(), lock.into()))
                .collect(),
        }
    }
}

#[pymethods]
impl PyKernelLocks {
    #[new]
    fn new(locks: BTreeMap<PyKernelDependency, PyKernelLock>) -> Self {
        Self { locks }
    }

    fn __len__(&self) -> usize {
        self.locks.len()
    }

    fn __getitem__(&self, dependency: &PyKernelDependency) -> PyResult<PyKernelLock> {
        self.locks
            .get(dependency)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(dependency.__repr__()))
    }

    fn __contains__(&self, dependency: &PyBound<'_, PyAny>) -> bool {
        dependency
            .extract::<PyKernelDependency>()
            .is_ok_and(|dependency| self.locks.contains_key(&dependency))
    }

    fn __iter__(&self) -> PyKernelDependencyIterator {
        PyKernelDependencyIterator {
            dependencies: self.keys().into_iter(),
        }
    }

    /// Get the lock for `dependency`, or `default` if it is not locked.
    #[pyo3(signature = (dependency, default = None))]
    fn get(
        &self,
        dependency: &PyBound<'_, PyAny>,
        default: Option<PyKernelLock>,
    ) -> Option<PyKernelLock> {
        dependency
            .extract::<PyKernelDependency>()
            .ok()
            .and_then(|dependency| self.locks.get(&dependency).cloned())
            .or(default)
    }

    /// Get the locked dependencies.
    fn keys(&self) -> Vec<PyKernelDependency> {
        self.locks.keys().cloned().collect()
    }

    /// Get the kernel locks.
    fn values(&self) -> Vec<PyKernelLock> {
        self.locks.values().cloned().collect()
    }

    /// Get the (dependency, lock) pairs.
    fn items(&self) -> Vec<(PyKernelDependency, PyKernelLock)> {
        self.locks
            .iter()
            .map(|(dependency, lock)| (dependency.clone(), lock.clone()))
            .collect()
    }

    /// Parse a `KernelLocks` collection from a JSON string.
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(s: &str) -> PyResult<Self> {
        let locks: KernelLocks = serde_json::from_str(s)
            .map_err(|err| PyValueError::new_err(format!("Cannot parse KernelLocks: {err:#}")))?;
        Ok(locks.into())
    }

    /// Serialize the locks collection to a pretty-printed JSON string.
    fn to_json(&self) -> PyResult<String> {
        let locks: KernelLocks = self.clone().into();
        serde_json::to_string_pretty(&locks)
            .map_err(|err| PyValueError::new_err(format!("Cannot serialize KernelLocks: {err:#}")))
    }

    fn __repr__(&self) -> String {
        let locks = self
            .locks
            .iter()
            .map(|(dependency, lock)| format!("{}: {}", dependency.__repr__(), lock.__repr__()))
            .collect::<Vec<_>>()
            .join(", ");
        format!("KernelLocks({{{locks}}})")
    }
}

/// A locked kernel revision with the SRI hash of the Nix output path.
#[pyclass(name = "NixKernelLock", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyNixKernelLock {
    commit: Oid,
    hash: String,
}

impl From<NixKernelLock> for PyNixKernelLock {
    fn from(lock: NixKernelLock) -> Self {
        Self {
            commit: lock.commit,
            hash: lock.hash,
        }
    }
}

impl From<PyNixKernelLock> for NixKernelLock {
    fn from(lock: PyNixKernelLock) -> Self {
        Self {
            commit: lock.commit,
            hash: lock.hash,
        }
    }
}

#[pymethods]
impl PyNixKernelLock {
    #[new]
    fn new(commit: &str, hash: String) -> PyResult<Self> {
        Ok(Self {
            commit: parse_oid(commit)?,
            hash,
        })
    }

    #[getter]
    fn commit(&self) -> &str {
        self.commit.as_str()
    }

    #[getter]
    fn hash(&self) -> &str {
        &self.hash
    }

    /// Parse a `NixKernelLock` from a JSON string.
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(s: &str) -> PyResult<Self> {
        let lock: NixKernelLock = serde_json::from_str(s)
            .map_err(|err| PyValueError::new_err(format!("Cannot parse NixKernelLock: {err:#}")))?;
        Ok(lock.into())
    }

    /// Serialize the lock to a pretty-printed JSON string.
    fn to_json(&self) -> PyResult<String> {
        let lock: NixKernelLock = self.clone().into();
        serde_json::to_string_pretty(&lock).map_err(|err| {
            PyValueError::new_err(format!("Cannot serialize NixKernelLock: {err:#}"))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "NixKernelLock(commit={:?}, hash={:?})",
            self.commit.as_str(),
            self.hash
        )
    }
}

/// Multiple (Nix) kernel locks keyed by the dependency they resolve.
///
///
/// This data structure is used to store lock files to be consumed
/// by nix-builder.
#[pyclass(name = "NixKernelLocks", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyNixKernelLocks {
    locks: BTreeMap<PyKernelDependency, PyNixKernelLock>,
}

impl From<NixKernelLocks> for PyNixKernelLocks {
    fn from(locks: NixKernelLocks) -> Self {
        Self {
            locks: locks
                .locks
                .into_iter()
                .map(|(dep, lock)| (dep.into(), lock.into()))
                .collect(),
        }
    }
}

impl From<PyNixKernelLocks> for NixKernelLocks {
    fn from(locks: PyNixKernelLocks) -> Self {
        Self {
            locks: locks
                .locks
                .into_iter()
                .map(|(dep, lock)| (dep.into(), lock.into()))
                .collect(),
        }
    }
}

#[pymethods]
impl PyNixKernelLocks {
    #[new]
    fn new(locks: BTreeMap<PyKernelDependency, PyNixKernelLock>) -> Self {
        Self { locks }
    }

    fn __len__(&self) -> usize {
        self.locks.len()
    }

    fn __getitem__(&self, dependency: &PyKernelDependency) -> PyResult<PyNixKernelLock> {
        self.locks
            .get(dependency)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(dependency.__repr__()))
    }

    fn __contains__(&self, dependency: &PyBound<'_, PyAny>) -> bool {
        dependency
            .extract::<PyKernelDependency>()
            .is_ok_and(|dependency| self.locks.contains_key(&dependency))
    }

    fn __iter__(&self) -> PyKernelDependencyIterator {
        PyKernelDependencyIterator {
            dependencies: self.keys().into_iter(),
        }
    }

    /// Get the lock for `dependency`, or `default` if it is not locked.
    #[pyo3(signature = (dependency, default = None))]
    fn get(
        &self,
        dependency: &PyBound<'_, PyAny>,
        default: Option<PyNixKernelLock>,
    ) -> Option<PyNixKernelLock> {
        dependency
            .extract::<PyKernelDependency>()
            .ok()
            .and_then(|dependency| self.locks.get(&dependency).cloned())
            .or(default)
    }

    /// Get the locked dependencies.
    fn keys(&self) -> Vec<PyKernelDependency> {
        self.locks.keys().cloned().collect()
    }

    /// Get the kernel locks.
    fn values(&self) -> Vec<PyNixKernelLock> {
        self.locks.values().cloned().collect()
    }

    /// Get the (dependency, lock) pairs.
    fn items(&self) -> Vec<(PyKernelDependency, PyNixKernelLock)> {
        self.locks
            .iter()
            .map(|(dependency, lock)| (dependency.clone(), lock.clone()))
            .collect()
    }

    /// Parse a `NixKernelLocks` collection from a JSON string.
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(s: &str) -> PyResult<Self> {
        let locks: NixKernelLocks = serde_json::from_str(s).map_err(|err| {
            PyValueError::new_err(format!("Cannot parse NixKernelLocks: {err:#}"))
        })?;
        Ok(locks.into())
    }

    /// Serialize the locks collection to a pretty-printed JSON string.
    fn to_json(&self) -> PyResult<String> {
        let locks: NixKernelLocks = self.clone().into();
        serde_json::to_string_pretty(&locks).map_err(|err| {
            PyValueError::new_err(format!("Cannot serialize NixKernelLocks: {err:#}"))
        })
    }

    fn __repr__(&self) -> String {
        let locks = self
            .locks
            .iter()
            .map(|(dependency, lock)| format!("{}: {}", dependency.__repr__(), lock.__repr__()))
            .collect::<Vec<_>>()
            .join(", ");
        format!("NixKernelLocks({{{locks}}})")
    }
}

/// A collection of kernel paths keyed by the dependency they resolve.
#[pyclass(name = "KernelPaths", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyKernelPaths {
    paths: BTreeMap<PyKernelDependency, PathBuf>,
}

impl From<KernelPaths> for PyKernelPaths {
    fn from(paths: KernelPaths) -> Self {
        Self {
            paths: paths
                .paths
                .into_iter()
                .map(|(dep, path)| (dep.into(), path))
                .collect(),
        }
    }
}

impl From<PyKernelPaths> for KernelPaths {
    fn from(paths: PyKernelPaths) -> Self {
        Self {
            paths: paths
                .paths
                .into_iter()
                .map(|(dep, path)| (dep.into(), path))
                .collect(),
        }
    }
}

#[pymethods]
impl PyKernelPaths {
    #[new]
    fn new(paths: BTreeMap<PyKernelDependency, PathBuf>) -> Self {
        Self { paths }
    }

    fn __len__(&self) -> usize {
        self.paths.len()
    }

    fn __getitem__(&self, dependency: &PyKernelDependency) -> PyResult<PathBuf> {
        self.paths
            .get(dependency)
            .cloned()
            .ok_or_else(|| PyKeyError::new_err(dependency.__repr__()))
    }

    fn __contains__(&self, dependency: &PyBound<'_, PyAny>) -> bool {
        dependency
            .extract::<PyKernelDependency>()
            .is_ok_and(|dependency| self.paths.contains_key(&dependency))
    }

    fn __iter__(&self) -> PyKernelDependencyIterator {
        PyKernelDependencyIterator {
            dependencies: self.keys().into_iter(),
        }
    }

    /// Get the path for `dependency`, or `default` if it has no path.
    #[pyo3(signature = (dependency, default = None))]
    fn get(&self, dependency: &PyBound<'_, PyAny>, default: Option<PathBuf>) -> Option<PathBuf> {
        dependency
            .extract::<PyKernelDependency>()
            .ok()
            .and_then(|dependency| self.paths.get(&dependency).cloned())
            .or(default)
    }

    /// Get the dependencies.
    fn keys(&self) -> Vec<PyKernelDependency> {
        self.paths.keys().cloned().collect()
    }

    /// Get the kernel paths.
    fn values(&self) -> Vec<PathBuf> {
        self.paths.values().cloned().collect()
    }

    /// Get the (dependency, path) pairs.
    fn items(&self) -> Vec<(PyKernelDependency, PathBuf)> {
        self.paths
            .iter()
            .map(|(dependency, path)| (dependency.clone(), path.clone()))
            .collect()
    }

    /// Parse a `KernelPaths` collection from a JSON string.
    #[staticmethod]
    #[pyo3(name = "from_json")]
    fn py_from_json(s: &str) -> PyResult<Self> {
        let paths: KernelPaths = serde_json::from_str(s)
            .map_err(|err| PyValueError::new_err(format!("Cannot parse KernelPaths: {err:#}")))?;
        Ok(paths.into())
    }

    /// Serialize the paths collection to a pretty-printed JSON string.
    fn to_json(&self) -> PyResult<String> {
        let paths: KernelPaths = self.clone().into();
        serde_json::to_string_pretty(&paths)
            .map_err(|err| PyValueError::new_err(format!("Cannot serialize KernelPaths: {err:#}")))
    }

    fn __repr__(&self) -> String {
        let paths = self
            .paths
            .iter()
            .map(|(dependency, path)| format!("{}: {:?}", dependency.__repr__(), path))
            .collect::<Vec<_>>()
            .join(", ");
        format!("KernelPaths({{{paths}}})")
    }
}
