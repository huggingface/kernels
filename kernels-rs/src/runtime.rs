use std::path::{Path, PathBuf};

use walkdir::WalkDir;

use crate::error::{Error, Result};

// Locate `libtvm_ffi.so` on the system in the following order
//
//  1. `$TVM_FFI_LIB` environment variable
//  2. Python `tvm_ffi` package via `python3 -c ...`
//  3. Recursive search under `$HOME` and `/nix/store`
//  4. Bare name (delegate to the dynamic linker / `LD_LIBRARY_PATH`)
pub fn find_tvm_ffi() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("TVM_FFI_LIB") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Some(path) = find_tvm_ffi_via_python() {
        return Ok(path);
    }

    let home = std::env::var("HOME").unwrap_or_default();
    for root in [home.as_str(), "/nix/store"] {
        if let Some(path) = find_file(Path::new(root), "libtvm_ffi.so") {
            return Ok(path);
        }
    }

    Ok(PathBuf::from("libtvm_ffi.so"))
}

fn find_tvm_ffi_via_python() -> Option<PathBuf> {
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            "import tvm_ffi, pathlib; print(pathlib.Path(tvm_ffi.__file__).parent)",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let pkg_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let so = PathBuf::from(&pkg_dir).join("lib/libtvm_ffi.so");
    so.exists().then_some(so)
}

fn find_file(root: &Path, target: &str) -> Option<PathBuf> {
    WalkDir::new(root)
        .max_depth(10)
        .into_iter()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy() == target)
        .map(|e| e.into_path())
}

pub fn load_tvm_ffi() -> Result<libloading::os::unix::Library> {
    let path = find_tvm_ffi()?;
    unsafe {
        libloading::os::unix::Library::open(Some(&path), libc::RTLD_NOW | libc::RTLD_GLOBAL)
            .map_err(|e| Error::Kernel(format!("Cannot load libtvm_ffi.so from {path:?}: {e}")))
    }
}

pub fn load_kernel(so_path: &Path) -> Result<libloading::Library> {
    unsafe { libloading::Library::new(so_path).map_err(Into::into) }
}

pub fn find_kernel_so(dir: &Path) -> Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('_') && name.ends_with(".so") {
            return Ok(entry.path());
        }
    }
    Err(Error::Kernel(format!(
        "no kernel .so found in {}",
        dir.display()
    )))
}

pub fn cache_dir() -> PathBuf {
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache/huggingface")
        })
        .join("kernels")
}
