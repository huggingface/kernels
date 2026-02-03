use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use kernel_abi_check::{
    MacOSViolation, ManylinuxViolation, PythonAbiViolation, Version, check_macos, check_manylinux,
};
use object::{BinaryFormat, Object as ObjectTrait};
use pyo3::Bound as PyBound;
use pyo3::exceptions::PyIOError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Object file that can be validated.
#[pyclass(name = "ObjectFile")]
struct PyObjectFile {
    filename: PathBuf,
    data: Vec<u8>,
}

impl PyObjectFile {
    fn parse_file(&self) -> PyResult<object::File<'_>> {
        object::File::parse(&*self.data).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot parse object file `{}`: {}",
                self.filename.to_string_lossy(),
                err
            ))
        })
    }
}

#[pymethods]
impl PyObjectFile {
    /// Create a new `ObjectFile` from a path.
    #[new]
    fn new(filename: PathBuf) -> PyResult<Self> {
        let data = fs::read(&filename).map_err(|err| {
            PyIOError::new_err(format!(
                "Cannot open object file `{}`: {}",
                filename.to_string_lossy(),
                err
            ))
        })?;

        Ok(Self { filename, data })
    }

    /// Check Python ABI compatibility for this object file  
    fn check_python_abi(&self, abi_version: String, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let file = self.parse_file()?;

        let python_abi = Version::from_str(&abi_version).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot parse Python ABI version `{abi_version}`: {err}",
            ))
        })?;

        let violations =
            kernel_abi_check::check_python_abi(&python_abi, file.format(), file.symbols())
                .map_err(|err| {
                    PyValueError::new_err(format!(
                        "Cannot check Python ABI for `{}`: {}",
                        self.filename.to_string_lossy(),
                        err
                    ))
                })?;

        let mut result = Vec::new();
        for violation in violations {
            let py_violation: Py<PyAny> = match violation {
                PythonAbiViolation::IncompatibleAbi3Symbol { name, added } => {
                    Py::new(py, PyIncompatibleAbi3Symbol { name, added })?.into()
                }
                PythonAbiViolation::NonAbi3Symbol { name } => {
                    Py::new(py, PyNonAbi3Symbol { name })?.into()
                }
            };
            result.push(py_violation);
        }
        Ok(result)
    }

    /// Check manylinux compatibility for this object file
    fn check_manylinux(&self, manylinux_version: String, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let file = self.parse_file()?;

        let violations = check_manylinux(
            &manylinux_version,
            file.architecture(),
            file.endianness(),
            file.symbols(),
        )
        .map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot check manylinux for `{}`: {}",
                self.filename.to_string_lossy(),
                err
            ))
        })?;

        let mut result = Vec::new();
        for violation in violations {
            let py_violation: Py<PyAny> = match violation {
                ManylinuxViolation::Symbol { name, dep, version } => {
                    Py::new(py, PyIncompatibleManylinuxSymbol { name, dep, version })?.into()
                }
            };
            result.push(py_violation);
        }
        Ok(result)
    }

    /// Check macOS compatibility for this object file
    fn check_macos(&self, macos_version: String, py: Python) -> PyResult<Vec<Py<PyAny>>> {
        let file = self.parse_file()?;

        let macos_ver = Version::from_str(&macos_version).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot parse macOS version `{macos_version}`: {err}",
            ))
        })?;

        let violations = check_macos(&file, &macos_ver).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot check macOS for `{}`: {}",
                self.filename.to_string_lossy(),
                err
            ))
        })?;

        let mut result = Vec::new();
        for violation in violations {
            let py_violation: Py<PyAny> = match violation {
                MacOSViolation::MissingMinOS => Py::new(py, PyMissingMacOSVersion)?.into(),
                MacOSViolation::IncompatibleMinOS { version } => {
                    Py::new(py, PyIncompatibleMacOSVersion { version })?.into()
                }
            };
            result.push(py_violation);
        }
        Ok(result)
    }

    /// Get the binary format of this object file
    fn format(&self) -> PyResult<PyBinaryFormat> {
        let file = self.parse_file()?;
        let binary_format = file.format();

        let py_format = match binary_format {
            BinaryFormat::Coff => PyBinaryFormat::Coff,
            BinaryFormat::Elf => PyBinaryFormat::Elf,
            BinaryFormat::MachO => PyBinaryFormat::MachO,
            BinaryFormat::Pe => PyBinaryFormat::Pe,
            BinaryFormat::Wasm => PyBinaryFormat::Wasm,
            BinaryFormat::Xcoff => PyBinaryFormat::Xcoff,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported binary format: {binary_format:?}"
                )));
            }
        };

        Ok(py_format)
    }
}

/// Binary format of an object file
#[derive(Clone, Debug, Eq, PartialEq)]
#[pyclass(name = "BinaryFormat")]
pub enum PyBinaryFormat {
    /// COFF (Common Object File Format)
    #[pyo3(name = "COFF")]
    Coff,
    /// ELF (Executable and Linkable Format)
    #[pyo3(name = "ELF")]
    Elf,
    /// Mach-O (Mach Object file format)
    #[pyo3(name = "MACH_O")]
    MachO,
    /// PE (Portable Executable)
    #[pyo3(name = "PE")]
    Pe,
    /// WebAssembly
    #[pyo3(name = "WASM")]
    Wasm,
    /// XCOFF (Extended Common Object File Format)
    #[pyo3(name = "XCOFF")]
    Xcoff,
}

#[pymethods]
impl PyBinaryFormat {
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __ne__(&self, other: &Self) -> bool {
        self != other
    }

    fn __repr__(&self) -> String {
        match self {
            PyBinaryFormat::Coff => "BinaryFormat.COFF".to_string(),
            PyBinaryFormat::Elf => "BinaryFormat.ELF".to_string(),
            PyBinaryFormat::MachO => "BinaryFormat.MACH_O".to_string(),
            PyBinaryFormat::Pe => "BinaryFormat.PE".to_string(),
            PyBinaryFormat::Wasm => "BinaryFormat.WASM".to_string(),
            PyBinaryFormat::Xcoff => "BinaryFormat.XCOFF".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Incompatible ABI3 symbol violation
#[derive(Clone, Debug, Eq, PartialEq)]
#[pyclass(name = "IncompatibleAbi3Symbol")]
struct PyIncompatibleAbi3Symbol {
    name: String,
    added: Version,
}

#[pymethods]
impl PyIncompatibleAbi3Symbol {
    #[new]
    #[pyo3(signature = (*py_args, name, added))]
    fn new(py_args: &Bound<'_, PyTuple>, name: String, added: String) -> PyResult<Self> {
        if !py_args.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "All arguments must be provided as keyword arguments",
            ));
        }
        let added = Version::from_str(&added).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot parse the version the symbol was added `{added}`: {err}",
            ))
        })?;
        Ok(Self { name, added })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn version_added(&self) -> String {
        self.added.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __ne__(&self, other: &Self) -> bool {
        self != other
    }

    fn __repr__(&self) -> String {
        format!(
            "IncompatibleAbi3Symbol(name='{}', version_added='{}')",
            self.name, self.added
        )
    }
}

/// Non-ABI3 symbol violation
#[pyclass(name = "NonAbi3Symbol")]
struct PyNonAbi3Symbol {
    name: String,
}

#[pymethods]
impl PyNonAbi3Symbol {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __repr__(&self) -> String {
        format!("NonAbi3Symbol(name='{}')", self.name)
    }
}

/// Manylinux symbol violation
#[derive(Clone, Debug, Eq, PartialEq)]
#[pyclass(name = "IncompatibleManylinuxSymbol")]
struct PyIncompatibleManylinuxSymbol {
    name: String,
    dep: String,
    version: String,
}

#[pymethods]
impl PyIncompatibleManylinuxSymbol {
    #[new]
    #[pyo3(signature = (*py_args, name, dep, version))]
    fn new(
        py_args: &Bound<'_, PyTuple>,
        name: String,
        dep: String,
        version: String,
    ) -> PyResult<Self> {
        if !py_args.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "All arguments must be provided as keyword arguments",
            ));
        }
        Ok(Self { name, dep, version })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn dep(&self) -> &str {
        &self.dep
    }

    #[getter]
    fn version(&self) -> &str {
        &self.version
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __ne__(&self, other: &Self) -> bool {
        self != other
    }

    fn __repr__(&self) -> String {
        format!(
            "IncompatibleManylinuxSymbol(name='{}', dep='{}', version='{}')",
            self.name, self.dep, self.version
        )
    }
}

/// Missing minimum OS version violation
#[derive(Clone, Debug, Eq, PartialEq)]
#[pyclass(name = "MissingMacOSVersion")]
struct PyMissingMacOSVersion;

#[pymethods]
impl PyMissingMacOSVersion {
    #[new]
    pub fn new() -> Self {
        Self
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __ne__(&self, other: &Self) -> bool {
        self != other
    }

    fn __repr__(&self) -> String {
        "MissingMacOSVersion()".to_string()
    }
}

/// Incompatible minimum OS version violation
#[derive(Clone, Debug, Eq, PartialEq)]
#[pyclass(name = "IncompatibleMacOSVersion")]
struct PyIncompatibleMacOSVersion {
    version: Version,
}

#[pymethods]
impl PyIncompatibleMacOSVersion {
    #[new]
    #[pyo3(signature = (*py_args, version))]
    fn new(py_args: &Bound<'_, PyTuple>, version: String) -> PyResult<Self> {
        if !py_args.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "All arguments must be provided as keyword arguments",
            ));
        }
        let version = Version::from_str(&version).map_err(|err| {
            PyValueError::new_err(format!(
                "Cannot parse the version the symbol was added `{version}`: {err}",
            ))
        })?;
        Ok(Self { version })
    }

    #[getter]
    fn version(&self) -> String {
        self.version.to_string()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __ne__(&self, other: &Self) -> bool {
        self != other
    }

    fn __repr__(&self) -> String {
        format!("IncompatibleMacOSVersion(version='{}')", self.version)
    }
}

#[pyo3::pymodule(name = "kernel_abi_check")]
fn kernel_abi_check_py(m: &PyBound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObjectFile>()?;

    // Binary format enum
    m.add_class::<PyBinaryFormat>()?;

    // Python ABI violation classes
    m.add_class::<PyIncompatibleAbi3Symbol>()?;
    m.add_class::<PyNonAbi3Symbol>()?;

    // Manylinux violation classes
    m.add_class::<PyIncompatibleManylinuxSymbol>()?;

    // macOS violation classes
    m.add_class::<PyMissingMacOSVersion>()?;
    m.add_class::<PyIncompatibleMacOSVersion>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
