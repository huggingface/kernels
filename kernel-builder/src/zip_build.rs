use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter},
    path::{Path, PathBuf},
};

use eyre::{bail, ensure, Context, Result};
use itertools::Itertools;
use kernels_common::config::{Build, BuildLayout};
use kernels_common::metadata::Metadata;
use walkdir::WalkDir;
use zip::{write::SimpleFileOptions, CompressionMethod, DateTime, ZipWriter};

use crate::util::{check_or_infer_kernel_dir, discover_variants};

/// Fixed deflate level for reproducibility.
const COMPRESSION_LEVEL: i64 = 6;

/// Fixed file permissions for reproducibility.
const FILE_PERMISSIONS: u32 = 0o644;

/// Extensions of native libraries that cannot be loaded from a ZIP file.
const NATIVE_EXTENSIONS: [&str; 5] = ["so", "dylib", "dll", "pyd", "metallib"];

/// Zip build variants that have "zip" as the build variant.
pub fn zip_build(source_dir: PathBuf, variants_dir: Option<PathBuf>) -> Result<()> {
    let source_dir = check_or_infer_kernel_dir(Some(source_dir))?;
    let variants_dir =
        check_or_infer_kernel_dir(Some(variants_dir.unwrap_or_else(|| source_dir.clone())))?;

    let build = Build::open(&source_dir)?;
    if build.general.build_layout != BuildLayout::Zip {
        return Ok(());
    }

    let (_, variants) = discover_variants(&variants_dir)?;

    for variant in variants {
        let variant_name = variant.file_name().unwrap().to_string_lossy();

        let metadata_path = variant.join("metadata.json");
        let f = File::open(&metadata_path).context(format!(
            "Cannot open `{}` for reading",
            metadata_path.to_string_lossy()
        ))?;
        let mut metadata: Metadata = serde_json::from_reader(BufReader::new(f))?;

        // The metadata reflects the state of the build, so there is nothing
        // to do if the variant was already zipped.
        if metadata.build_layout == BuildLayout::Zip {
            continue;
        }

        eprintln!("Zipping variant `{variant_name}`...");
        zip_variant(&variant, &metadata_path, &mut metadata)?;
    }

    Ok(())
}

/// Returns true for files that should not be included in a ZIP archive.
fn is_excluded(variant: &Path, path: &Path) -> bool {
    let relative = path.strip_prefix(variant).unwrap_or(path);

    // Metadata holds the digest, so must be readable without opening the
    // archive. The signature is typically not generated yet when zipping
    // build variants, but let's be defensive.
    if relative == Path::new("metadata.json") || relative == Path::new("metadata.json.sigstore") {
        return true;
    }

    // Exclude interpreter bytecode.
    if relative
        .components()
        .any(|c| c.as_os_str() == "__pycache__")
        || path.extension().and_then(|e| e.to_str()) == Some("pyc")
    {
        return true;
    }

    // Paths are validated to be UTF-8 by the caller, so this is not lossy.
    let file_name = path
        .file_name()
        .map(|f| f.to_string_lossy())
        .unwrap_or_default();

    // Exclude dynamic libraries.
    NATIVE_EXTENSIONS.iter().any(|ext| {
        file_name.ends_with(&format!(".{ext}")) || file_name.contains(&format!(".{ext}."))
    })
}

fn zip_variant(variant: &Path, metadata_path: &Path, metadata: &mut Metadata) -> Result<()> {
    let archive_path = variant.join(format!("{}.zip", metadata.id));
    ensure!(
        !archive_path.exists(),
        "`{}` already exists",
        archive_path.to_string_lossy()
    );

    // Created sorted list (to avoid filesystem order influencing
    // reproducibility) of files to be zipped.
    let mut files = Vec::new();
    for entry in WalkDir::new(variant).sort_by_file_name() {
        let entry = entry.wrap_err("Failed to read directory entry for zipping")?;
        let file_type = entry.file_type();
        if file_type.is_dir() {
            continue;
        }
        if !file_type.is_file() {
            bail!(
                "Cannot zip `{}`: not a regular file",
                entry.path().to_string_lossy()
            );
        }
        // Entry names must be UTF-8. Fail when one is not.
        ensure!(
            entry.path().to_str().is_some(),
            "Cannot zip `{}`: path is not valid UTF-8",
            entry.path().to_string_lossy()
        );
        if is_excluded(variant, entry.path()) {
            continue;
        }
        files.push(entry.into_path());
    }

    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(COMPRESSION_LEVEL))
        .last_modified_time(DateTime::default())
        .unix_permissions(FILE_PERMISSIONS);

    let archive = File::create(&archive_path).context(format!(
        "Cannot create `{}`",
        archive_path.to_string_lossy()
    ))?;
    let mut writer = ZipWriter::new(BufWriter::new(archive));

    for path in &files {
        let name = path
            .strip_prefix(variant)?
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        writer.start_file(name, options)?;
        let mut file = File::open(path)
            .wrap_err_with(|| format!("Cannot open `{}` for zipping", path.display()))?;
        io::copy(&mut file, &mut writer)?;
    }
    writer.finish()?;

    // Update the metadata to indicate the ZIP layout is used.
    metadata.set_kernel_layout_to_zip();
    let f = File::create(metadata_path).context(format!(
        "Cannot open `{}` for writing",
        metadata_path.to_string_lossy()
    ))?;
    serde_json::to_writer_pretty(BufWriter::new(f), metadata).context(format!(
        "Cannot write updated metadata to `{}`",
        metadata_path.to_string_lossy()
    ))?;

    // Remove zipped files only after the archive was written successfully.
    for path in &files {
        fs::remove_file(path).wrap_err_with(|| format!("Cannot remove `{}`", path.display()))?;
    }

    // Remove bytecode directories. This is needed to make the empty
    // directory removal complete.
    remove_bytecode_dirs(variant)?;
    remove_empty_dirs(variant)?;

    Ok(())
}

fn remove_bytecode_dirs(root: &Path) -> Result<()> {
    let dirs = WalkDir::new(root)
        .into_iter()
        .filter_ok(|e| e.file_type().is_dir() && e.file_name() == "__pycache__")
        .map_ok(|e| e.into_path())
        .collect::<Result<Vec<_>, _>>()?;
    for dir in dirs.into_iter().filter(|d| d.exists()) {
        fs::remove_dir_all(&dir).wrap_err_with(|| format!("Cannot remove `{}`", dir.display()))?;
    }
    Ok(())
}

/// Remove empty directories below `root`, but not `root` itself.
fn remove_empty_dirs(root: &Path) -> Result<()> {
    for entry in WalkDir::new(root).contents_first(true).min_depth(1) {
        let entry = entry?;
        if entry.file_type().is_dir() && fs::read_dir(entry.path())?.next().is_none() {
            fs::remove_dir(entry.path())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use kernels_common::config::BuildLayout;
    use tempfile::TempDir;
    use zip::ZipArchive;

    use super::*;

    fn write_kernel(kernel: &Path, layout: &str) -> PathBuf {
        fs::write(
            kernel.join("build.toml"),
            format!(
                r#"
[general]
name = "test-kernel"
version = 1
edition = 6
license = "Apache-2.0"
build-layout = "{layout}"
backends = ["cuda"]

[torch-noarch]
"#
            ),
        )
        .unwrap();

        let dir = kernel.join("build/torch-cuda");
        fs::create_dir_all(dir.join("pkg/layers")).unwrap();
        fs::create_dir_all(dir.join("pkg/__pycache__")).unwrap();
        fs::write(
            dir.join("metadata.json"),
            r#"{"name":"test-kernel","id":"test-id","version":1,"kernels-minver":"0.14.0","license":"apache-2.0","python-depends":[],"backend":{"type":"cuda"}}"#,
        )
        .unwrap();
        fs::write(dir.join("pkg/__init__.py"), b"import _ops").unwrap();
        fs::write(dir.join("pkg/_ops.py"), b"ops").unwrap();
        fs::write(dir.join("pkg/layers/__init__.py"), b"layers").unwrap();
        fs::write(dir.join("pkg/data.json"), b"{}").unwrap();
        fs::write(dir.join("pkg/_ext.abi3.so"), b"elf").unwrap();
        fs::write(dir.join("pkg/libfoo.so.1"), b"elf").unwrap();
        fs::write(dir.join("pkg/kernels.metallib"), b"metal").unwrap();
        fs::write(dir.join("pkg/__pycache__/_ops.cpython-312.pyc"), b"pyc").unwrap();
        dir
    }

    fn read_metadata(variant: &Path) -> Metadata {
        serde_json::from_reader(File::open(variant.join("metadata.json")).unwrap()).unwrap()
    }

    fn archive_names(path: &Path) -> Vec<String> {
        let mut archive = ZipArchive::new(File::open(path).unwrap()).unwrap();
        (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().unwrap().to_string())
            .collect()
    }

    #[test]
    fn files_layout_is_untouched() {
        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "files");

        zip_build(kernel.path().to_path_buf(), None).unwrap();

        assert!(!variant.join("test-id.zip").exists());
        assert!(variant.join("pkg/_ops.py").exists());
        assert_eq!(read_metadata(&variant).build_layout, BuildLayout::Files);
    }

    #[test]
    fn zip_layout_zips_everything_but_native_files() {
        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "zip");

        zip_build(kernel.path().to_path_buf(), None).unwrap();

        assert_eq!(
            archive_names(&variant.join("test-id.zip")),
            [
                "pkg/__init__.py",
                "pkg/_ops.py",
                "pkg/data.json",
                "pkg/layers/__init__.py",
            ]
        );

        // Zipped files and empty directories are gone, the rest stays.
        assert!(!variant.join("pkg/_ops.py").exists());
        assert!(!variant.join("pkg/layers").exists());
        assert!(variant.join("metadata.json").exists());
        assert!(variant.join("pkg/_ext.abi3.so").exists());
        assert!(variant.join("pkg/libfoo.so.1").exists());
        assert!(variant.join("pkg/kernels.metallib").exists());
        assert!(!variant.join("pkg/__pycache__").exists());

        let mut archive =
            ZipArchive::new(File::open(variant.join("test-id.zip")).unwrap()).unwrap();
        let mut content = String::new();
        archive
            .by_name("pkg/_ops.py")
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert_eq!(content, "ops");
    }

    #[test]
    fn zip_updates_metadata() {
        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "zip");
        assert_eq!(read_metadata(&variant).build_layout, BuildLayout::Files);

        zip_build(kernel.path().to_path_buf(), None).unwrap();

        let metadata = read_metadata(&variant);
        assert_eq!(metadata.build_layout, BuildLayout::Zip);
        assert_eq!(metadata.kernels_minver.unwrap().to_string(), "0.19.0");
    }

    #[test]
    fn variants_can_be_in_a_different_directory() {
        let source = TempDir::new().unwrap();
        let output = TempDir::new().unwrap();
        write_kernel(source.path(), "zip");
        // Move the variant to the output directory, which has no `build.toml`.
        let variant = output.path().join("torch-cuda");
        fs::rename(source.path().join("build/torch-cuda"), &variant).unwrap();

        zip_build(
            source.path().to_path_buf(),
            Some(output.path().to_path_buf()),
        )
        .unwrap();

        assert!(variant.join("test-id.zip").exists());
        assert_eq!(read_metadata(&variant).build_layout, BuildLayout::Zip);
    }

    #[test]
    fn zip_build_is_idempotent() {
        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "zip");

        zip_build(kernel.path().to_path_buf(), None).unwrap();
        let archive = fs::read(variant.join("test-id.zip")).unwrap();
        let metadata = fs::read(variant.join("metadata.json")).unwrap();

        // Variants that were already zipped are skipped.
        zip_build(kernel.path().to_path_buf(), None).unwrap();
        assert_eq!(fs::read(variant.join("test-id.zip")).unwrap(), archive);
        assert_eq!(fs::read(variant.join("metadata.json")).unwrap(), metadata);
    }

    #[test]
    fn existing_archive_for_unzipped_variant_is_an_error() {
        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "zip");
        fs::write(variant.join("test-id.zip"), b"not an archive").unwrap();

        assert!(zip_build(kernel.path().to_path_buf(), None).is_err());
        assert_eq!(read_metadata(&variant).build_layout, BuildLayout::Files);
        assert!(variant.join("pkg/_ops.py").exists());
    }

    #[test]
    fn zip_is_reproducible() {
        let kernel1 = TempDir::new().unwrap();
        let kernel2 = TempDir::new().unwrap();
        let variant1 = write_kernel(kernel1.path(), "zip");
        let variant2 = write_kernel(kernel2.path(), "zip");

        zip_build(kernel1.path().to_path_buf(), None).unwrap();
        // Different modification time must not change the archive.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        zip_build(kernel2.path().to_path_buf(), None).unwrap();

        assert_eq!(
            fs::read(variant1.join("test-id.zip")).unwrap(),
            fs::read(variant2.join("test-id.zip")).unwrap()
        );
    }

    // macOS file systems reject file names that are not valid UTF-8, so such
    // a file cannot be created there.
    #[cfg(target_os = "linux")]
    #[test]
    fn non_utf8_path_is_an_error() {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

        let kernel = TempDir::new().unwrap();
        let variant = write_kernel(kernel.path(), "zip");
        fs::write(
            variant.join("pkg").join(OsStr::from_bytes(b"lib\xff.so")),
            b"elf",
        )
        .unwrap();

        assert!(zip_build(kernel.path().to_path_buf(), None).is_err());
        assert!(!variant.join("test-id.zip").exists());
        assert!(variant.join("pkg/_ops.py").exists());
        assert_eq!(read_metadata(&variant).build_layout, BuildLayout::Files);
    }
}
