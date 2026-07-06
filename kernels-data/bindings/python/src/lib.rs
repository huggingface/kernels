use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;

use kernels_data::config::{Backend, KernelName};
use kernels_data::digest::{Digest, DigestAlgorithm, DigestViolation};
use kernels_data::metadata::{BackendInfo, BuildInfo, GitHash, KernelBuilderInfo, Metadata};
use kernels_data::version::Version;
use pyo3::Bound as PyBound;
use pyo3::exceptions::{PyException, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// A dotted numeric version (e.g. `12.8.0`). Trailing zeros are stripped
/// during normalization.
#[pyclass(name = "Version", frozen, eq, ord, hash)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PyVersion {
    inner: Version,
}

#[pymethods]
impl PyVersion {
    /// Parse a version string of the form `X`, `X.Y`, `X.Y.Z`, ...
    #[staticmethod]
    #[pyo3(name = "from_str")]
    fn py_from_str(s: &str) -> PyResult<Self> {
        Version::from_str(s)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(format!("Cannot parse version `{s}`: {err}")))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Version('{}')", self.inner)
    }
}

/// A validated kernel name matching `^[a-z][-a-z0-9]*[a-z0-9]$`.
#[pyclass(name = "KernelName", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PyKernelName {
    inner: KernelName,
}

#[pymethods]
impl PyKernelName {
    #[new]
    fn new(name: String) -> PyResult<Self> {
        KernelName::new(name)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("KernelName('{}')", self.inner)
    }

    /// The kernel name with dashes replaced by underscores, suitable for
    /// use as a Python identifier.
    #[getter]
    fn python_name(&self) -> String {
        self.inner.python_name()
    }
}

/// Kernel backend (hardware target).
#[pyclass(name = "Backend", eq, frozen, hash)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PyBackend {
    #[pyo3(name = "CANN")]
    Cann,
    #[pyo3(name = "CPU")]
    Cpu,
    #[pyo3(name = "CUDA")]
    Cuda,
    #[pyo3(name = "Metal")]
    Metal,
    #[pyo3(name = "Neuron")]
    Neuron,
    #[pyo3(name = "ROCm")]
    Rocm,
    #[pyo3(name = "XPU")]
    Xpu,
}

impl From<Backend> for PyBackend {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Cann => PyBackend::Cann,
            Backend::Cpu => PyBackend::Cpu,
            Backend::Cuda => PyBackend::Cuda,
            Backend::Metal => PyBackend::Metal,
            Backend::Neuron => PyBackend::Neuron,
            Backend::Rocm => PyBackend::Rocm,
            Backend::Xpu => PyBackend::Xpu,
        }
    }
}

impl From<PyBackend> for Backend {
    fn from(b: PyBackend) -> Self {
        match b {
            PyBackend::Cann => Backend::Cann,
            PyBackend::Cpu => Backend::Cpu,
            PyBackend::Cuda => Backend::Cuda,
            PyBackend::Metal => Backend::Metal,
            PyBackend::Neuron => Backend::Neuron,
            PyBackend::Rocm => Backend::Rocm,
            PyBackend::Xpu => Backend::Xpu,
        }
    }
}

#[pymethods]
impl PyBackend {
    /// Parse a backend name (`"cann"`, `"cpu"`, `"cuda"`, `"metal"`,
    /// `"neuron"`, `"rocm"`, `"xpu"`).
    #[staticmethod]
    #[pyo3(name = "from_str")]
    fn py_from_str(s: &str) -> PyResult<Self> {
        Backend::from_str(s)
            .map(Into::into)
            .map_err(PyValueError::new_err)
    }

    fn __str__(&self) -> &'static str {
        Backend::from(*self).as_str()
    }

    fn __repr__(&self) -> String {
        let variant = match self {
            PyBackend::Cann => "CANN",
            PyBackend::Cpu => "CPU",
            PyBackend::Cuda => "CUDA",
            PyBackend::Metal => "Metal",
            PyBackend::Neuron => "Neuron",
            PyBackend::Rocm => "ROCm",
            PyBackend::Xpu => "XPU",
        };
        format!("Backend.{variant}")
    }
}

/// Backend information
#[pyclass(name = "BackendInfo", frozen)]
#[derive(Clone, Debug)]
struct PyBackendInfo {
    backend_type: PyBackend,
    archs: Option<Vec<String>>,
}

impl From<BackendInfo> for PyBackendInfo {
    fn from(backend_info: BackendInfo) -> Self {
        Self {
            backend_type: backend_info.backend_type.into(),
            archs: backend_info.archs,
        }
    }
}

#[pymethods]
impl PyBackendInfo {
    fn __repr__(&self) -> String {
        format!(
            "BackendInfo(backend_type={}, archs={:?})",
            self.backend_type.__repr__(),
            self.archs
        )
    }

    #[getter]
    fn backend_type(&self) -> PyBackend {
        self.backend_type
    }

    #[getter]
    fn archs(&self) -> Option<&[String]> {
        self.archs.as_deref()
    }
}

#[pyclass(name = "GitHash", frozen)]
#[derive(Clone, Debug)]
struct PyGitHash {
    sha: String,
    dirty: bool,
}

impl From<GitHash> for PyGitHash {
    fn from(g: GitHash) -> Self {
        Self {
            sha: g.sha,
            dirty: g.dirty,
        }
    }
}

#[pymethods]
impl PyGitHash {
    #[getter]
    fn sha(&self) -> &str {
        &self.sha
    }

    #[getter]
    fn dirty(&self) -> bool {
        self.dirty
    }

    fn __repr__(&self) -> String {
        format!("GitHash(sha={:?}, dirty={})", self.sha, self.dirty)
    }
}

#[pyclass(name = "KernelBuilderInfo", frozen)]
#[derive(Clone, Debug)]
struct PyKernelBuilderInfo {
    version: String,
    git: Option<PyGitHash>,
}

impl From<KernelBuilderInfo> for PyKernelBuilderInfo {
    fn from(kb: KernelBuilderInfo) -> Self {
        Self {
            version: kb.version,
            git: kb.git.map(Into::into),
        }
    }
}

#[pymethods]
impl PyKernelBuilderInfo {
    #[getter]
    fn version(&self) -> &str {
        &self.version
    }

    /// Commit SHA + dirty state of the `kernel-builder` source, when known.
    #[getter]
    fn git(&self) -> Option<PyGitHash> {
        self.git.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelBuilderInfo(version={:?}, git={})",
            self.version,
            self.git
                .as_ref()
                .map_or("None".to_string(), |g| g.__repr__())
        )
    }
}

#[pyclass(name = "BuildInfo", frozen)]
#[derive(Clone, Debug)]
struct PyBuildInfo {
    kernel_builder: Option<PyKernelBuilderInfo>,
    kernel: Option<PyGitHash>,
}

impl From<BuildInfo> for PyBuildInfo {
    fn from(b: BuildInfo) -> Self {
        Self {
            kernel_builder: b.kernel_builder.map(Into::into),
            kernel: b.kernel.map(Into::into),
        }
    }
}

#[pymethods]
impl PyBuildInfo {
    #[getter]
    fn kernel_builder(&self) -> Option<PyKernelBuilderInfo> {
        self.kernel_builder.clone()
    }

    #[getter]
    fn kernel(&self) -> Option<PyGitHash> {
        self.kernel.clone()
    }

    /// Whether either the `kernel-builder` or the kernel source was dirty.
    #[getter]
    fn dirty(&self) -> bool {
        self.kernel_builder
            .as_ref()
            .and_then(|kb| kb.git.as_ref())
            .is_some_and(|g| g.dirty)
            || self.kernel.as_ref().is_some_and(|k| k.dirty)
    }

    fn __repr__(&self) -> String {
        format!(
            "BuildInfo(kernel_builder={}, kernel={})",
            self.kernel_builder
                .as_ref()
                .map_or("None".to_string(), |kb| kb.__repr__()),
            self.kernel
                .as_ref()
                .map_or("None".to_string(), |k| k.__repr__())
        )
    }
}

/// Parsed `metadata.json` for a kernel build variant.
#[pyclass(name = "Metadata", frozen)]
#[derive(Clone, Debug)]
struct PyMetadata {
    id: String,
    name: PyKernelName,
    version: usize,
    license: String,
    upstream: Option<String>,
    source: Option<String>,
    python_depends: Vec<String>,
    backend: PyBackendInfo,
    digest: Option<PyDigest>,
    build_info: Option<PyBuildInfo>,
}

impl From<Metadata> for PyMetadata {
    fn from(m: Metadata) -> Self {
        Self {
            id: m.id,
            name: PyKernelName { inner: m.name },
            version: m.version,
            license: m.license,
            upstream: m.upstream.map(|u| u.as_url().to_string()),
            source: m.source.map(|u| u.as_url().to_string()),
            python_depends: m.python_depends,
            backend: m.backend.into(),
            digest: m.digest.map(Into::into),
            build_info: m.build_info.map(Into::into),
        }
    }
}

#[pymethods]
impl PyMetadata {
    /// Parse `metadata.json` at the given path.
    ///
    /// Raises `ValueError` on any I/O or parse error.
    #[staticmethod]
    fn read_from_file(metadata_path: PathBuf) -> PyResult<Self> {
        let f = File::open(&metadata_path).map_err(|err| {
            PyOSError::new_err(format!("Failed to open `{metadata_path:?}`: {err:#}"))
        })?;
        Metadata::from_reader(BufReader::new(f))
            .map(Into::into)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "Cannot parse metadata from `{metadata_path:?}`: {err:#}"
                ))
            })
    }

    /// Parse `metadata.json` from JSON in a byte array.
    ///
    /// Raises `ValueError` on any parse error.
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        Metadata::from_bytes(bytes)
            .map(Into::into)
            .map_err(|err| PyValueError::new_err(format!("Cannot parse metadata: {err:#}")))
    }

    #[getter]
    fn id(&self) -> &str {
        &self.id
    }

    #[getter]
    fn name(&self) -> PyKernelName {
        self.name.clone()
    }

    #[getter]
    fn version(&self) -> usize {
        self.version
    }

    #[getter]
    fn license(&self) -> &str {
        &self.license
    }

    #[getter]
    fn upstream(&self) -> Option<&str> {
        self.upstream.as_deref()
    }

    #[getter]
    fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }

    #[getter]
    fn python_depends(&self) -> &[String] {
        &self.python_depends
    }

    #[getter]
    fn backend(&self) -> PyBackendInfo {
        self.backend.clone()
    }

    #[getter]
    fn digest(&self) -> Option<PyDigest> {
        self.digest.clone()
    }

    #[getter]
    fn build_info(&self) -> Option<PyBuildInfo> {
        self.build_info.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(id={}, name={:?}, version={:?}, license={:?}, upstream={:?}, source={:?}, python_depends={:?}, backend={}, digest={}, build_info={})",
            self.id,
            self.name,
            self.version,
            self.license,
            self.upstream,
            self.source,
            self.python_depends,
            self.backend.__repr__(),
            self.digest
                .as_ref()
                .map_or("None".to_string(), |sd| sd.__repr__()),
            self.build_info
                .as_ref()
                .map_or("None".to_string(), |bi| bi.__repr__())
        )
    }
}

/// A violation of a digest when validated against a reference digest.
///
/// This tagged union covers the types of violations. Each violation can be
/// converted to a string using ``str(violation)``.
#[pyclass(name = "DigestViolation")]
#[derive(Clone)]
enum PyDigestViolation {
    MissingFile {
        path: String,
    },
    UnknownFile {
        path: String,
    },
    HashMismatch {
        path: String,
        expected: String,
        got: String,
    },
    AlgorithmMismatch {
        expected: PyDigestAlgorithm,
        got: PyDigestAlgorithm,
    },
}

/// Digest algorithm.
#[pyclass(name = "DigestAlgorithm", frozen, eq, hash)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum PyDigestAlgorithm {
    #[pyo3(name = "SHA256")]
    Sha256,
    #[pyo3(name = "SHA512")]
    Sha512,
}

impl From<DigestAlgorithm> for PyDigestAlgorithm {
    fn from(a: DigestAlgorithm) -> Self {
        match a {
            DigestAlgorithm::SHA256 => Self::Sha256,
            DigestAlgorithm::SHA512 => Self::Sha512,
        }
    }
}

impl From<PyDigestAlgorithm> for DigestAlgorithm {
    fn from(a: PyDigestAlgorithm) -> Self {
        match a {
            PyDigestAlgorithm::Sha256 => DigestAlgorithm::SHA256,
            PyDigestAlgorithm::Sha512 => DigestAlgorithm::SHA512,
        }
    }
}

#[pymethods]
impl PyDigestAlgorithm {
    fn __str__(&self) -> &'static str {
        match self {
            Self::Sha256 => "sha256",
            Self::Sha512 => "sha512",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::Sha256 => "DigestAlgorithm.SHA256",
            Self::Sha512 => "DigestAlgorithm.SHA512",
        }
    }
}

impl From<DigestViolation> for PyDigestViolation {
    fn from(v: DigestViolation) -> Self {
        match v {
            DigestViolation::MissingFile { path } => Self::MissingFile { path },
            DigestViolation::UnknownFile { path } => Self::UnknownFile { path },
            DigestViolation::HashMismatch {
                path,
                expected,
                got,
            } => Self::HashMismatch {
                path,
                expected,
                got,
            },
            DigestViolation::AlgorithmMismatch { expected, got } => Self::AlgorithmMismatch {
                expected: expected.into(),
                got: got.into(),
            },
        }
    }
}

impl From<PyDigestViolation> for DigestViolation {
    fn from(v: PyDigestViolation) -> Self {
        match v {
            PyDigestViolation::MissingFile { path } => Self::MissingFile { path },
            PyDigestViolation::UnknownFile { path } => Self::UnknownFile { path },
            PyDigestViolation::HashMismatch {
                path,
                expected,
                got,
            } => Self::HashMismatch {
                path,
                expected,
                got,
            },
            PyDigestViolation::AlgorithmMismatch { expected, got } => Self::AlgorithmMismatch {
                expected: expected.into(),
                got: got.into(),
            },
        }
    }
}

#[pymethods]
impl PyDigestViolation {
    // Delegate to the core `Display` impl so the message formatting lives in a
    // single place.
    fn __str__(&self) -> String {
        DigestViolation::from(self.clone()).to_string()
    }
}

pyo3::create_exception!(
    kernels_data,
    DigestValidationError,
    PyException,
    "Raised by `Digest.validate` when the actual digest does not match the \
     reference digest.\n\n\
     The string representation lists every violation. The individual violations \
     are also available as a list of `DigestViolation` via the `violations` \
     attribute."
);

/// Digest for a kernel build variant.
#[pyclass(name = "Digest", frozen)]
#[derive(Clone, Debug)]
struct PyDigest {
    inner: Digest,
}

impl From<Digest> for PyDigest {
    fn from(inner: Digest) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDigest {
    /// Hash the files in `variant_path` using `algorithm`.
    #[staticmethod]
    fn hash_variant(algorithm: PyDigestAlgorithm, variant_path: PathBuf) -> PyResult<PyDigest> {
        match Digest::hash_variant(algorithm.into(), &variant_path) {
            Ok(digest) => Ok(digest.into()),
            Err(err) => {
                let msg = format!(
                    "Failed to hash variant `{}`: {err:#}",
                    variant_path.display()
                );
                let is_io = err
                    .chain()
                    .any(|e| e.downcast_ref::<std::io::Error>().is_some());
                if is_io {
                    Err(PyOSError::new_err(msg))
                } else {
                    Err(PyRuntimeError::new_err(msg))
                }
            }
        }
    }

    /// Validate `other` against this digest.
    ///
    /// Raises `DigestValidationError` if the digests do not match.
    fn validate(&self, py: Python<'_>, other: &PyDigest) -> PyResult<()> {
        match self.inner.validate(&other.inner) {
            Ok(()) => Ok(()),
            Err(err) => {
                let violations = err
                    .violations()
                    .iter()
                    .cloned()
                    .map(PyDigestViolation::from)
                    .collect::<Vec<_>>();

                // Build the exception instance with the rendered message as its
                // single argument (so `str(exc)` lists every violation), and
                // expose the structured violations via a `violations` attribute.
                let instance = py
                    .get_type::<DigestValidationError>()
                    .call1((err.to_string(),))?;
                instance.setattr("violations", violations)?;
                Err(PyErr::from_value(instance))
            }
        }
    }

    #[getter]
    fn algorithm(&self) -> PyDigestAlgorithm {
        self.inner.algorithm().into()
    }

    #[getter]
    fn files(&self) -> BTreeMap<String, String> {
        self.inner.files().clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Digest(algorithm={}, files={:?})",
            self.algorithm().__repr__(),
            self.inner.files()
        )
    }
}

#[pyo3::pymodule(name = "kernels_data")]
fn kernels_data_py(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyBackendInfo>()?;
    m.add_class::<PyBuildInfo>()?;
    m.add_class::<PyGitHash>()?;
    m.add_class::<PyKernelBuilderInfo>()?;
    m.add_class::<PyKernelName>()?;
    m.add_class::<PyMetadata>()?;
    m.add_class::<PyVersion>()?;
    m.add_class::<PyDigestAlgorithm>()?;
    m.add_class::<PyDigest>()?;
    m.add_class::<PyDigestViolation>()?;
    m.add(
        "DigestValidationError",
        m.py().get_type::<DigestValidationError>(),
    )?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
