use std::path::PathBuf;

use kernels_common::signing::receipt::{
    DigestReceipt, DigestReceiptStore, KernelLocation, Receipt, ReceiptStoreError,
    SignatureReceipt, SignatureReceiptStore,
};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use crate::git::PyOid;

pyo3::create_exception!(
    _rust,
    ReceiptError,
    PyException,
    "Raised by `SignatureReceiptStore` and `DigestReceiptStore` when a \
     receipt cannot be read, written, or interpreted.\n\n\
     A missing receipt is not an error: `load` returns `None` for it. Since \
     a receipt is only a cache of a previous verification, callers can treat \
     this exception as a cache miss and re-verify, at the cost of not \
     noticing a cache that is persistently broken."
);

fn receipt_error(err: ReceiptStoreError) -> PyErr {
    ReceiptError::new_err(format!("{:#}", eyre::Report::new(err)))
}

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

/// Receipt of a successful signature verification of a kernel.
#[pyclass(name = "SignatureReceipt", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PySignatureReceipt {
    inner: SignatureReceipt,
}

impl From<SignatureReceipt> for PySignatureReceipt {
    fn from(inner: SignatureReceipt) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySignatureReceipt {
    #[new]
    fn new(location: &PyKernelLocation) -> Self {
        SignatureReceipt::new(location.inner.clone()).into()
    }

    #[getter]
    fn location(&self) -> PyKernelLocation {
        self.inner.location().clone().into()
    }

    fn __repr__(&self) -> String {
        format!("SignatureReceipt(location={})", self.location().__repr__())
    }
}

/// Store of kernel signature verification receipts.
#[pyclass(name = "SignatureReceiptStore", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PySignatureReceiptStore {
    inner: SignatureReceiptStore,
}

#[pymethods]
impl PySignatureReceiptStore {
    /// The signature receipt store inside the kernels cache.
    ///
    /// Raises `ReceiptError` when the cache directory cannot be determined,
    /// in which case verifications cannot be cached.
    #[staticmethod]
    fn in_kernels_cache() -> PyResult<Self> {
        SignatureReceiptStore::in_kernels_cache()
            .map(|inner| PySignatureReceiptStore { inner })
            .map_err(receipt_error)
    }

    /// A signature receipt store in the given directory.
    #[staticmethod]
    fn from_path(path: PathBuf) -> Self {
        PySignatureReceiptStore {
            inner: SignatureReceiptStore::from_path(path),
        }
    }

    /// The receipt for `location`, or `None` when the kernel signature has
    /// not been verified yet.
    ///
    /// Raises `ReceiptError` if a receipt exists but cannot be used.
    fn load(&self, location: &PyKernelLocation) -> PyResult<Option<PySignatureReceipt>> {
        self.inner
            .load(&location.inner)
            .map(|receipt| receipt.map(Into::into))
            .map_err(receipt_error)
    }

    /// Store `receipt`, replacing any existing receipt for its location.
    ///
    /// Raises `ReceiptError` if the receipt cannot be written.
    fn store(&self, receipt: &PySignatureReceipt) -> PyResult<()> {
        self.inner.store(&receipt.inner).map_err(receipt_error)
    }
}

/// Receipt of a successful digest verification of a kernel.
#[pyclass(name = "DigestReceipt", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PyDigestReceipt {
    inner: DigestReceipt,
}

impl From<DigestReceipt> for PyDigestReceipt {
    fn from(inner: DigestReceipt) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDigestReceipt {
    #[new]
    fn new(location: &PyKernelLocation) -> Self {
        DigestReceipt::new(location.inner.clone()).into()
    }

    #[getter]
    fn location(&self) -> PyKernelLocation {
        self.inner.location().clone().into()
    }

    fn __repr__(&self) -> String {
        format!("DigestReceipt(location={})", self.location().__repr__())
    }
}

/// Store of kernel digest verification receipts.
#[pyclass(name = "DigestReceiptStore", frozen)]
#[derive(Clone, Debug)]
pub(crate) struct PyDigestReceiptStore {
    inner: DigestReceiptStore,
}

#[pymethods]
impl PyDigestReceiptStore {
    /// The digest receipt store inside the kernels cache.
    ///
    /// Raises `ReceiptError` when the cache directory cannot be determined,
    /// in which case verifications cannot be cached.
    #[staticmethod]
    fn in_kernels_cache() -> PyResult<Self> {
        DigestReceiptStore::in_kernels_cache()
            .map(|inner| PyDigestReceiptStore { inner })
            .map_err(receipt_error)
    }

    /// A digest receipt store in the given directory.
    #[staticmethod]
    fn from_path(path: PathBuf) -> Self {
        PyDigestReceiptStore {
            inner: DigestReceiptStore::from_path(path),
        }
    }

    /// The receipt for `location`, or `None` when the kernel digest has not
    /// been verified yet.
    ///
    /// Raises `ReceiptError` if a receipt exists but cannot be used.
    fn load(&self, location: &PyKernelLocation) -> PyResult<Option<PyDigestReceipt>> {
        self.inner
            .load(&location.inner)
            .map(|receipt| receipt.map(Into::into))
            .map_err(receipt_error)
    }

    /// Store `receipt`, replacing any existing receipt for its location.
    ///
    /// Raises `ReceiptError` if the receipt cannot be written.
    fn store(&self, receipt: &PyDigestReceipt) -> PyResult<()> {
        self.inner.store(&receipt.inner).map_err(receipt_error)
    }
}
