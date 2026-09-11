use std::path::PathBuf;

use kernels_common::signing::receipt::{KernelLocation, ReceiptStore, VerificationReceipt};
use pyo3::exceptions::{PyException, PyOSError};
use pyo3::prelude::*;

pyo3::create_exception!(
    _rust,
    ReceiptError,
    PyException,
    "Raised by `ReceiptStore` when a receipt cannot be read, written, or \
     interpreted.\n\n\
     A missing receipt is not an error: `ReceiptStore.load` returns `None` \
     for it. Since a receipt is only a cache of a previous verification, \
     callers can treat this exception as a cache miss and re-verify, at the \
     cost of not noticing a cache that is persistently broken."
);

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
    /// The location of a kernel variant in a local directory.
    ///
    /// Fingerprints the variant's files, so that the location changes when
    /// the kernel is rebuilt or re-signed.
    #[staticmethod]
    fn local(variant_path: PathBuf) -> PyResult<Self> {
        KernelLocation::local(&variant_path)
            .map(Into::into)
            .map_err(|err| {
                PyOSError::new_err(format!(
                    "Cannot fingerprint variant `{}`: {err:#}",
                    variant_path.display()
                ))
            })
    }

    /// The location of a kernel variant in a Hub repository.
    #[staticmethod]
    fn remote(repo_id: String, revision: String, variant: String) -> Self {
        KernelLocation::remote(repo_id, revision, variant).into()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            KernelLocation::LocalKernel { variant_path, .. } => {
                format!("KernelLocation.local(variant_path={variant_path:?})")
            }
            KernelLocation::RemoteKernel {
                repo_id,
                revision,
                variant,
            } => format!(
                "KernelLocation.remote(repo_id={repo_id:?}, revision={revision:?}, variant={variant:?})"
            ),
        }
    }
}

/// Receipt of a successful kernel verification.
#[pyclass(name = "VerificationReceipt", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyVerificationReceipt {
    inner: VerificationReceipt,
}

impl From<VerificationReceipt> for PyVerificationReceipt {
    fn from(inner: VerificationReceipt) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVerificationReceipt {
    #[new]
    fn new(location: &PyKernelLocation) -> Self {
        VerificationReceipt::new(location.inner.clone()).into()
    }

    #[getter]
    fn location(&self) -> PyKernelLocation {
        self.inner.location().clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "VerificationReceipt(location={})",
            self.location().__repr__()
        )
    }
}

/// Store of kernel verification receipts.
#[pyclass(name = "ReceiptStore", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyReceiptStore {
    inner: ReceiptStore,
}

#[pymethods]
impl PyReceiptStore {
    /// The receipt store inside the kernels cache.
    ///
    /// Raises `ReceiptError` when the cache directory cannot be determined,
    /// in which case verifications cannot be cached.
    #[staticmethod]
    fn in_kernels_cache() -> PyResult<Self> {
        ReceiptStore::in_kernels_cache()
            .map(|inner| PyReceiptStore { inner })
            .map_err(|err| ReceiptError::new_err(format!("{:#}", eyre::Report::new(err))))
    }

    /// A receipt store in the given directory.
    #[staticmethod]
    fn from_path(path: PathBuf) -> Self {
        PyReceiptStore {
            inner: ReceiptStore::from_path(path),
        }
    }

    /// The receipt for `location`, or `None` when the kernel has not been
    /// verified yet.
    ///
    /// Raises `ReceiptError` if a receipt exists but cannot be used.
    fn load(&self, location: &PyKernelLocation) -> PyResult<Option<PyVerificationReceipt>> {
        self.inner
            .load(&location.inner)
            .map(|receipt| receipt.map(Into::into))
            .map_err(|err| ReceiptError::new_err(format!("{:#}", eyre::Report::new(err))))
    }

    /// Store `receipt`, replacing any existing receipt for its location.
    ///
    /// Raises `ReceiptError` if the receipt cannot be written.
    fn store(&self, receipt: &PyVerificationReceipt) -> PyResult<()> {
        self.inner
            .store(&receipt.inner)
            .map_err(|err| ReceiptError::new_err(format!("{:#}", eyre::Report::new(err))))
    }
}
