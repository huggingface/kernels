use std::fmt;
use std::process::Command;
use std::str::FromStr;

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
    cuda_version_from_smi().or_else(cuda_version_from_nvcc)
}

fn cuda_version_from_smi() -> Option<String> {
    let output = Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let rest = stdout.split("CUDA Version:").nth(1)?;
    Some(rest.split_whitespace().next()?.to_string())
}

fn cuda_version_from_nvcc() -> Option<String> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let after = stdout.split("release ").nth(1)?;
    Some(after.split(',').next()?.trim().to_string())
}

pub fn detect() -> BackendKind {
    if detect_cuda_version().is_some() {
        BackendKind::Cuda
    } else {
        BackendKind::Cpu
    }
}
