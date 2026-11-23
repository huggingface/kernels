use std::path::{Path, PathBuf};

use huggingface_hub::blocking::HfApiSync;
use huggingface_hub::{ListRepoTreeParams, RepoTreeEntry};

use crate::backend::{self, Backend, BackendKind};
use crate::error::{Error, Result};

#[derive(Debug)]
pub struct Variant {
    pub dir: PathBuf,
    pub backend: Backend,
}

fn parse(name: &str, dir: PathBuf) -> Option<Variant> {
    let rest = name.strip_prefix("tvm-ffi")?;
    let tag = rest.split('-').nth(1)?;

    let backend = match tag {
        "cpu" => Backend::Cpu,
        _ if tag.starts_with("cu") => {
            let raw = tag.strip_prefix("cu")?;
            if raw.len() < 2 {
                return None;
            }
            let (major, minor) = raw.split_at(raw.len() - 1);
            Backend::Cuda {
                version: format!("{major}.{minor}"),
            }
        }
        _ if tag.starts_with("xpu") => Backend::Xpu {
            version: tag.strip_prefix("xpu")?.to_string(),
        },
        _ => return None,
    };

    Some(Variant { dir, backend })
}

pub fn discover_local(base: &Path) -> Vec<Variant> {
    let Ok(entries) = std::fs::read_dir(base) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter_map(|e| parse(&e.file_name().to_string_lossy(), e.path()))
        .collect()
}

pub fn discover_remote(api: &HfApiSync, repo_id: &str, revision: &str) -> Result<Vec<Variant>> {
    let params = ListRepoTreeParams::builder()
        .repo_id(repo_id)
        .revision(revision)
        .recursive(true)
        .build();

    let entries = api.list_repo_tree(&params)?;

    Ok(entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::Directory { path, .. } => {
                let name = path.strip_prefix("build/")?;
                if name.contains('/') {
                    return None;
                }
                parse(name, PathBuf::from(name))
            }
            _ => None,
        })
        .collect())
}

// Pick the best variant for the given backend.
pub fn resolve(variants: &[Variant], kind: BackendKind) -> Result<&Variant> {
    match kind {
        BackendKind::Cpu => variants
            .iter()
            .find(|v| matches!(v.backend, Backend::Cpu))
            .ok_or_else(|| Error::Kernel("no CPU variant available".into())),

        BackendKind::Cuda => {
            let system_ver = backend::detect_cuda_version()
                .ok_or_else(|| Error::Kernel("CUDA requested but no runtime found".into()))?;

            let (sys_major, sys_minor) = parse_version(&system_ver);

            variants
                .iter()
                .filter_map(|v| match &v.backend {
                    Backend::Cuda { version } => {
                        let (major, minor) = parse_version(version);
                        if major == sys_major && minor <= sys_minor {
                            Some((v, minor))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .max_by_key(|(_, minor)| *minor)
                .map(|(v, _)| v)
                .ok_or_else(|| {
                    Error::Kernel(format!("no compatible CUDA variant for CUDA {system_ver}"))
                })
        }

        BackendKind::Xpu => variants
            .iter()
            .find(|v| matches!(v.backend, Backend::Xpu { .. }))
            .ok_or_else(|| Error::Kernel("no XPU variant available".into())),
    }
}

fn parse_version(ver: &str) -> (&str, u32) {
    let mut parts = ver.splitn(2, '.');
    let major = parts.next().unwrap_or("0");
    let minor: u32 = parts.next().unwrap_or("0").parse().unwrap_or(0);
    (major, minor)
}
