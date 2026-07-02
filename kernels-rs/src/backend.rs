use std::fmt;
use std::str::FromStr;

use libloading::Library;

use crate::error::Error;

#[derive(Debug, Clone)]
pub enum Backend {
    Cpu,
    Cuda { version: String },
    Xpu { version: String },
}

impl Backend {
    pub fn kind(&self) -> BackendKind {
        match self {
            Backend::Cpu => BackendKind::Cpu,
            Backend::Cuda { .. } => BackendKind::Cuda,
            Backend::Xpu { .. } => BackendKind::Xpu,
        }
    }

    pub fn name(&self) -> &str {
        self.kind().as_str()
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Cpu => write!(f, "cpu"),
            Backend::Cuda { version } => write!(f, "cuda {version}"),
            Backend::Xpu { version } => write!(f, "xpu {version}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Xpu,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Xpu => "xpu",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for BackendKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "xpu" => Ok(Self::Xpu),
            other => Err(Error::Kernel(format!("unknown backend: {other}"))),
        }
    }
}

pub fn detect_cuda_version() -> Option<String> {
    type CudaRuntimeGetVersion = unsafe extern "C" fn(*mut i32) -> i32;

    let library = unsafe { Library::new(libloading::library_filename("cudart")) }.ok()?;
    let cuda_runtime_get_version: libloading::Symbol<CudaRuntimeGetVersion> =
        unsafe { library.get(b"cudaRuntimeGetVersion\0").ok()? };

    let mut runtime_version = 0;
    if unsafe { cuda_runtime_get_version(&mut runtime_version) } != 0 {
        return None;
    }

    format_cuda_runtime_version(runtime_version)
}

fn format_cuda_runtime_version(runtime_version: i32) -> Option<String> {
    if runtime_version <= 0 {
        return None;
    }

    let major = runtime_version / 1000;
    let minor = (runtime_version % 1000) / 10;
    Some(format!("{major}.{minor}"))
}

pub fn detect() -> BackendKind {
    if detect_cuda_version().is_some() {
        BackendKind::Cuda
    } else {
        BackendKind::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::format_cuda_runtime_version;

    #[test]
    fn formats_cuda_runtime_versions() {
        assert_eq!(format_cuda_runtime_version(12080).as_deref(), Some("12.8"));
        assert_eq!(format_cuda_runtime_version(11020).as_deref(), Some("11.2"));
        assert_eq!(format_cuda_runtime_version(0), None);
    }
}
