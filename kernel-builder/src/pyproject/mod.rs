use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use eyre::{bail, Result};
use kernels_data::config::{Build, Framework};
use minijinja::Environment;

use crate::util::{check_or_infer_kernel_dir, check_or_infer_target_dir, parse_build};

pub(crate) mod common;
pub mod deps;
pub mod fileset;
mod kernel;
mod ops_identifier;
mod torch;
mod tvm_ffi;

pub use fileset::FileSet;
pub use kernels_data::metadata::parse_metadata;

pub fn create_pyproject_file_set(
    build: Build,
    target_dir: impl AsRef<Path>,
    ops_id: Option<String>,
) -> Result<FileSet> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    minijinja_embed::load_templates!(&mut env);

    let file_set = if matches!(build.framework, Framework::TvmFfi(_)) {
        tvm_ffi::write_tvm_ffi_ext(&env, &build, target_dir, ops_id)?
    } else if build.is_noarch() {
        torch::write_torch_ext_noarch(&env, &build, target_dir, ops_id)?
    } else {
        torch::write_torch_ext(&env, &build, target_dir, ops_id)?
    };

    Ok(file_set)
}

pub fn create_pyproject(
    kernel_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    force: bool,
    ops_id: Option<String>,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let target_dir = check_or_infer_target_dir(&kernel_dir, target_dir)?;
    let build = parse_build(&kernel_dir)?;
    let file_set = create_pyproject_file_set(build, &target_dir, ops_id)?;
    file_set.write(&target_dir, force)?;

    Ok(())
}

pub fn clean_pyproject(
    kernel_dir: Option<PathBuf>,
    target_dir: Option<PathBuf>,
    dry_run: bool,
    force: bool,
    ops_id: Option<String>,
) -> Result<()> {
    let kernel_dir = check_or_infer_kernel_dir(kernel_dir)?;
    let target_dir = check_or_infer_target_dir(&kernel_dir, target_dir)?;

    let build = parse_build(&kernel_dir)?;
    let generated_files =
        create_pyproject_file_set(build, target_dir.clone(), ops_id)?.into_names();

    if generated_files.is_empty() {
        eprintln!("No generated artifacts found to clean.");
        return Ok(());
    }

    if dry_run {
        println!("Files that would be deleted:");
        for file in &generated_files {
            if file.exists() {
                println!("  {}", file.to_string_lossy());
            }
        }
        return Ok(());
    }

    let existing_files: Vec<_> = generated_files.iter().filter(|f| f.exists()).collect();

    if existing_files.is_empty() {
        eprintln!("No generated artifacts found to clean.");
        return Ok(());
    }

    if !force {
        println!("Files to be deleted:");
        for file in &existing_files {
            println!("  {}", file.to_string_lossy());
        }
        print!("Continue? [y/N] ");
        std::io::stdout().flush()?;

        let mut response = String::new();
        std::io::stdin().read_line(&mut response)?;
        let response = response.trim().to_lowercase();

        if response != "y" && response != "yes" {
            eprintln!("Aborted.");
            return Ok(());
        }
    }

    let mut deleted_count = 0;
    let mut errors = Vec::new();

    for file in existing_files {
        match fs::remove_file(file) {
            Ok(_) => {
                deleted_count += 1;
                println!("Deleted: {}", file.to_string_lossy());
            }
            Err(e) => {
                errors.push(format!(
                    "Failed to delete {}: {}",
                    file.to_string_lossy(),
                    e
                ));
            }
        }
    }

    // Clean up empty directories
    let dirs_to_check = [target_dir.join("cmake")];

    for dir in dirs_to_check {
        if dir.exists() && is_empty_dir(&dir)? {
            match fs::remove_dir(&dir) {
                Ok(_) => println!("Removed empty directory: {}", dir.to_string_lossy()),
                Err(e) => eyre::bail!("Failed to remove directory `{}`: {e:?}", dir.display()),
            }
        }
    }

    if !errors.is_empty() {
        for error in errors {
            eprintln!("Error: {error}");
        }
        bail!("Some files could not be deleted");
    }

    println!("Cleaned {deleted_count} generated artifacts.");
    Ok(())
}

fn is_empty_dir(dir: &Path) -> Result<bool> {
    if !dir.is_dir() {
        return Ok(false);
    }

    let mut entries = fs::read_dir(dir)?;
    Ok(entries.next().is_none())
}
