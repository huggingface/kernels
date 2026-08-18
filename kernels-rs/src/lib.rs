pub mod backend;
#[cfg(feature = "candle")]
pub mod candle;
pub mod error;
pub mod runtime;
pub mod tvm_ffi;
pub mod variant;

use std::path::Path;

use huggingface_hub::blocking::HfApiSync;
use huggingface_hub::{DownloadFileParams, ListRepoTreeParams, RepoTreeEntry};

pub use backend::{Backend, BackendKind};
pub use error::{Error, Result};

pub struct KernelModule {
    _tvm_ffi: libloading::os::unix::Library,
    kernel: libloading::Library,
    backend: Backend,
}

impl KernelModule {
    fn load(so_path: &Path, backend: Backend) -> Result<Self> {
        let tvm_ffi = runtime::load_tvm_ffi()?;
        let kernel = runtime::load_kernel(so_path)?;
        Ok(Self {
            _tvm_ffi: tvm_ffi,
            kernel,
            backend,
        })
    }

    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// # Safety
    /// `symbol` must name a function with the `TVMFFIFunc` signature.
    pub unsafe fn get_func(&self, symbol: &[u8]) -> Result<tvm_ffi::TVMFFIFunc> {
        let sym: libloading::Symbol<tvm_ffi::TVMFFIFunc> = unsafe { self.kernel.get(symbol)? };
        Ok(*sym)
    }
}

pub fn get_kernel(repo_id: &str, version: u32) -> Result<KernelModule> {
    get_kernel_for_backend(repo_id, version, backend::detect())
}

pub fn get_kernel_for_backend(
    repo_id: &str,
    version: u32,
    kind: BackendKind,
) -> Result<KernelModule> {
    let api = HfApiSync::new().map_err(|e| Error::Kernel(format!("{e}")))?;
    let revision = format!("v{version}");

    let variants = variant::discover_remote(&api, repo_id, &revision)?;
    if variants.is_empty() {
        return Err(Error::Kernel(format!(
            "no build variants found in {repo_id} (revision: {revision})"
        )));
    }

    let variant = variant::resolve(&variants, kind)?;
    let variant_prefix = format!("build/{}", variant.dir.display());

    let files = collect_variant_files(&api, repo_id, &revision, &variant_prefix)?;
    if files.is_empty() {
        return Err(Error::Kernel(format!(
            "no files found under {variant_prefix} in {repo_id}"
        )));
    }

    let variant_dir = download_variant_files(&api, repo_id, &revision, &files)?;
    let so_path = runtime::find_kernel_so(&variant_dir)?;
    KernelModule::load(&so_path, variant.backend.clone())
}

pub fn get_local_kernel(repo_path: &Path) -> Result<KernelModule> {
    get_local_kernel_for_backend(repo_path, backend::detect())
}

pub fn get_local_kernel_for_backend(repo_path: &Path, kind: BackendKind) -> Result<KernelModule> {
    for base in [repo_path.to_path_buf(), repo_path.join("build")] {
        let variants = variant::discover_local(&base);
        if variants.is_empty() {
            continue;
        }
        let variant = variant::resolve(&variants, kind)?;
        let so_path = runtime::find_kernel_so(&variant.dir)?;
        return KernelModule::load(&so_path, variant.backend.clone());
    }

    Err(Error::Kernel(format!(
        "no kernel variants found in {}",
        repo_path.display()
    )))
}

fn collect_variant_files(
    api: &HfApiSync,
    repo_id: &str,
    revision: &str,
    prefix: &str,
) -> Result<Vec<String>> {
    let params = ListRepoTreeParams::builder()
        .repo_id(repo_id)
        .revision(revision)
        .recursive(true)
        .build();

    let entries = api.list_repo_tree(&params)?;

    Ok(entries
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, .. } if path.starts_with(prefix) => Some(path),
            _ => None,
        })
        .collect())
}

fn download_variant_files(
    api: &HfApiSync,
    repo_id: &str,
    revision: &str,
    files: &[String],
) -> Result<std::path::PathBuf> {
    let cache_dir = runtime::cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let mut variant_dir = None;
    for file in files {
        let params = DownloadFileParams::builder()
            .repo_id(repo_id)
            .filename(file)
            .local_dir(cache_dir.clone())
            .revision(revision)
            .build();

        let path = api.download_file(&params)?;

        if variant_dir.is_none() {
            variant_dir = path.parent().map(|p| p.to_path_buf());
        }
    }

    variant_dir.ok_or_else(|| Error::Kernel("Cannot determine local variant directory".into()))
}
