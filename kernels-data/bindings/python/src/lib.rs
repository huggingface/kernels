use std::path::PathBuf;
use std::str::FromStr;

use kernels_data::config::{Backend, KernelName};
use kernels_data::metadata::{Metadata, parse_metadata};
use kernels_data::version::Version;
use pyo3::Bound as PyBound;
use pyo3::exceptions::PyValueError;
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

/// Parsed `metadata.json` for a kernel build variant.
#[pyclass(name = "Metadata", frozen)]
#[derive(Clone, Debug)]
struct PyMetadata {
    version: Option<usize>,
    license: Option<String>,
    upstream: Option<String>,
    python_depends: Vec<String>,
    backend: PyBackend,
}

impl From<Metadata> for PyMetadata {
    fn from(m: Metadata) -> Self {
        Self {
            version: m.version,
            license: m.license,
            upstream: m.upstream.map(|u| u.to_string()),
            python_depends: m.python_depends,
            backend: m.backend.backend_type.into(),
        }
    }
}

#[pymethods]
impl PyMetadata {
    /// Parse `metadata.json` at the given path.
    ///
    /// Raises `ValueError` on any I/O or parse error.
    #[staticmethod]
    fn load(metadata_path: PathBuf) -> PyResult<Self> {
        parse_metadata(&metadata_path)
            .map(Into::into)
            .map_err(|err| PyValueError::new_err(format!("{err:#}")))
    }

    #[getter]
    fn version(&self) -> Option<usize> {
        self.version
    }

    #[getter]
    fn license(&self) -> Option<&str> {
        self.license.as_deref()
    }

    #[getter]
    fn upstream(&self) -> Option<&str> {
        self.upstream.as_deref()
    }

    #[getter]
    fn python_depends(&self) -> Vec<String> {
        self.python_depends.clone()
    }

    #[getter]
    fn backend(&self) -> PyBackend {
        self.backend
    }

    fn __repr__(&self) -> String {
        format!(
            "Metadata(version={:?}, license={:?}, upstream={:?}, python_depends={:?}, backend={})",
            self.version,
            self.license,
            self.upstream,
            self.python_depends,
            self.backend.__repr__()
        )
    }
}

/// Parse a kernel `metadata.json` file.
///
/// Raises `ValueError` on any I/O or parse error.
#[pyfunction(name = "parse_metadata")]
fn py_parse_metadata(path: PathBuf) -> PyResult<PyMetadata> {
    parse_metadata(&path)
        .map(Into::into)
        .map_err(|err| PyValueError::new_err(format!("{err:#}")))
}

#[pyo3::pymodule(name = "kernels_data")]
fn kernels_data_py(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyKernelName>()?;
    m.add_class::<PyMetadata>()?;
    m.add_class::<PyVersion>()?;
    m.add_function(wrap_pyfunction!(py_parse_metadata, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
