use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;

use kernels_data::config::{Backend, Build, General, KernelDependency, KernelName, KernelVersion};
use kernels_data::digest::{Digest, DigestAlgorithm, DigestViolation};
use kernels_data::lock::{KernelLock, KernelLocks};
use kernels_data::metadata::{BackendInfo, GitHash, KernelBuilderVersion, Metadata, Provenance};
use kernels_data::version::Version;
use pyo3::Bound as PyBound;
use pyo3::exceptions::{PyException, PyKeyError, PyOSError, PyRuntimeError, PyValueError};
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
    #[pyo3(name = "TPU")]
    Tpu,
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
            Backend::Tpu => PyBackend::Tpu,
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
            PyBackend::Tpu => Backend::Tpu,
            PyBackend::Xpu => Backend::Xpu,
        }
    }
}

#[pymethods]
impl PyBackend {
    /// Parse a backend name (`"cann"`, `"cpu"`, `"cuda"`, `"metal"`,
    /// `"neuron"`, `"rocm"`, `"tpu"`, `"xpu"`).
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
            PyBackend::Tpu => "TPU",
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

#[pyclass(name = "KernelBuilderVersion", frozen)]
#[derive(Clone, Debug)]
struct PyKernelBuilderVersion {
    version: String,
    git: Option<PyGitHash>,
}

impl From<KernelBuilderVersion> for PyKernelBuilderVersion {
    fn from(kb: KernelBuilderVersion) -> Self {
        Self {
            version: kb.version,
            git: kb.git.map(Into::into),
        }
    }
}

#[pymethods]
impl PyKernelBuilderVersion {
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
            "KernelBuilderVersion(version={:?}, git={})",
            self.version,
            self.git
                .as_ref()
                .map_or("None".to_string(), |g| g.__repr__())
        )
    }
}

#[pyclass(name = "Provenance", frozen)]
#[derive(Clone, Debug)]
struct PyProvenance {
    kernel_builder: PyKernelBuilderVersion,
    kernel: Option<PyGitHash>,
}

impl From<Provenance> for PyProvenance {
    fn from(b: Provenance) -> Self {
        Self {
            kernel_builder: b.kernel_builder.into(),
            kernel: b.kernel.map(Into::into),
        }
    }
}

#[pymethods]
impl PyProvenance {
    #[getter]
    fn kernel_builder(&self) -> PyKernelBuilderVersion {
        self.kernel_builder.clone()
    }

    #[getter]
    fn kernel(&self) -> Option<PyGitHash> {
        self.kernel.clone()
    }

    /// Whether either the `kernel-builder` or the kernel source was dirty.
    #[getter]
    fn dirty(&self) -> bool {
        self.kernel_builder.git.as_ref().is_some_and(|g| g.dirty)
            || self.kernel.as_ref().is_some_and(|k| k.dirty)
    }

    fn __repr__(&self) -> String {
        format!(
            "Provenance(kernel_builder={}, kernel={})",
            self.kernel_builder.__repr__(),
            self.kernel
                .as_ref()
                .map_or("None".to_string(), |k| k.__repr__())
        )
    }
}

/// A kernel version: either a numeric version or a git revision string.
#[pyclass(name = "KernelVersion", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum PyKernelVersion {
    Version { version: usize },
    Revision { revision: String },
}

impl From<KernelVersion> for PyKernelVersion {
    fn from(v: KernelVersion) -> Self {
        match v {
            KernelVersion::Version(n) => Self::Version { version: n },
            KernelVersion::Revision(s) => Self::Revision { revision: s },
        }
    }
}

impl From<PyKernelVersion> for KernelVersion {
    fn from(v: PyKernelVersion) -> Self {
        match v {
            PyKernelVersion::Version { version } => Self::Version(version),
            PyKernelVersion::Revision { revision } => Self::Revision(revision),
        }
    }
}

#[pymethods]
impl PyKernelVersion {
    fn __repr__(&self) -> String {
        match self {
            Self::Version { version } => format!("KernelVersion.Version(version={version})"),
            Self::Revision { revision } => {
                format!("KernelVersion.Revision(revision={revision:?})")
            }
        }
    }
}

/// A dependency on another kernel.
#[pyclass(name = "KernelDependency", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PyKernelDependency {
    repo_id: String,
    version: PyKernelVersion,
}

impl From<KernelDependency> for PyKernelDependency {
    fn from(d: KernelDependency) -> Self {
        Self {
            repo_id: d.repo_id,
            version: d.version.into(),
        }
    }
}

impl From<PyKernelDependency> for KernelDependency {
    fn from(d: PyKernelDependency) -> Self {
        Self {
            repo_id: d.repo_id,
            version: d.version.into(),
        }
    }
}

#[pymethods]
impl PyKernelDependency {
    #[new]
    fn new(repo_id: String, version: PyKernelVersion) -> Self {
        Self { repo_id, version }
    }

    #[getter]
    fn repo_id(&self) -> &str {
        &self.repo_id
    }

    #[getter]
    fn version(&self) -> PyKernelVersion {
        self.version.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelDependency(repo_id={:?}, version={})",
            self.repo_id,
            self.version.__repr__()
        )
    }
}

/// A locked kernel revision and its transitive dependencies.
#[pyclass(name = "KernelLock", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PyKernelLock {
    repo_id: String,
    revision: String,
    depends: PyKernelLocks,
}

impl From<KernelLock> for PyKernelLock {
    fn from(lock: KernelLock) -> Self {
        Self {
            repo_id: lock.repo_id,
            revision: lock.revision,
            depends: lock.depends.into(),
        }
    }
}

impl From<PyKernelLock> for KernelLock {
    fn from(lock: PyKernelLock) -> Self {
        Self {
            repo_id: lock.repo_id,
            revision: lock.revision,
            depends: lock.depends.into(),
        }
    }
}

#[pymethods]
impl PyKernelLock {
    #[new]
    fn new(repo_id: String, revision: String, depends: PyKernelLocks) -> Self {
        Self {
            repo_id,
            revision,
            depends,
        }
    }

    #[getter]
    fn repo_id(&self) -> &str {
        &self.repo_id
    }

    #[getter]
    fn revision(&self) -> &str {
        &self.revision
    }

    #[getter]
    fn depends(&self) -> PyKernelLocks {
        self.depends.clone()
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
        format!(
            "KernelLock(repo_id={:?}, revision={:?}, depends={})",
            self.repo_id,
            self.revision,
            self.depends.__repr__()
        )
    }
}

/// A collection of locked kernels keyed by the dependency they resolve.
///
/// Behaves as a read-only mapping from [`PyKernelDependency`] to
/// [`PyKernelLock`]. The map is ordered, so iteration is deterministic.
#[pyclass(name = "KernelLocks", frozen, eq, hash)]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PyKernelLocks {
    locks: BTreeMap<PyKernelDependency, PyKernelLock>,
}

/// Iterator over the dependencies of a [`PyKernelLocks`].
#[pyclass(name = "KernelLocksIterator")]
struct PyKernelLocksIterator {
    dependencies: std::vec::IntoIter<PyKernelDependency>,
}

#[pymethods]
impl PyKernelLocksIterator {
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

    fn __iter__(&self) -> PyKernelLocksIterator {
        PyKernelLocksIterator {
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

/// General kernel configuration common to all backends.
#[pyclass(name = "General", frozen)]
#[derive(Clone, Debug)]
struct PyGeneral {
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
struct PyBuild {
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
    kernel_depends: Vec<PyKernelDependency>,
    backend: PyBackendInfo,
    digest: Option<PyDigest>,
    provenance: Option<PyProvenance>,
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
            kernel_depends: m.kernel_depends.into_iter().map(Into::into).collect(),
            backend: m.backend.into(),
            digest: m.digest.map(Into::into),
            provenance: m.provenance.map(Into::into),
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
    fn kernel_depends(&self) -> Vec<PyKernelDependency> {
        self.kernel_depends.clone()
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
    fn provenance(&self) -> Option<PyProvenance> {
        self.provenance.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(id={}, name={:?}, version={:?}, license={:?}, upstream={:?}, source={:?}, python_depends={:?}, kernel_depends={:?}, backend={}, digest={}, provenance={})",
            self.id,
            self.name,
            self.version,
            self.license,
            self.upstream,
            self.source,
            self.python_depends,
            self.kernel_depends,
            self.backend.__repr__(),
            self.digest
                .as_ref()
                .map_or("None".to_string(), |sd| sd.__repr__()),
            self.provenance
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
    m.add_class::<PyProvenance>()?;
    m.add_class::<PyGitHash>()?;
    m.add_class::<PyKernelBuilderVersion>()?;
    m.add_class::<PyKernelName>()?;
    m.add_class::<PyKernelVersion>()?;
    m.add_class::<PyKernelDependency>()?;
    m.add_class::<PyKernelLock>()?;
    m.add_class::<PyKernelLocks>()?;
    m.add_class::<PyGeneral>()?;
    m.add_class::<PyBuild>()?;
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
