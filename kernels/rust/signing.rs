use kernels_common::signing::receipt::KernelLocation;
use pyo3::prelude::*;

use crate::git::PyOid;

/// The location of a kernel that a verification applies to.
#[pyclass(name = "KernelLocation", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyKernelLocation {
    inner: KernelLocation,
}

impl From<KernelLocation> for PyKernelLocation {
    fn from(inner: KernelLocation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKernelLocation {
    /// The location of a kernel variant in a Hub repository.
    #[staticmethod]
    fn remote(repo_id: String, revision: PyOid, variant: String) -> Self {
        KernelLocation::remote(repo_id, revision.into_inner(), variant).into()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            KernelLocation::RemoteKernel {
                repo_id,
                revision,
                variant,
            } => format!(
                "KernelLocation.remote(repo_id={:?}, revision={:?}, variant={:?})",
                repo_id,
                revision.as_str(),
                variant
            ),
        }
    }
}
