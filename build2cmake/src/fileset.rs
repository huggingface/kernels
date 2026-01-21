use std::collections::HashMap;
use std::path::{Path, PathBuf};

use eyre::{bail, eyre, Context, Result};

pub struct FileSet(HashMap<PathBuf, Vec<u8>>);

impl FileSet {
    pub fn new() -> FileSet {
        FileSet(HashMap::new())
    }

    fn check_exist(&self, target_dir: &Path) -> Result<()> {
        let mut existing = Vec::new();
        for path in self.0.keys() {
            let full_path = target_dir.join(path);
            if full_path.exists() {
                existing.push(path.to_string_lossy().into_owned());
            }
        }

        if !existing.is_empty() {
            bail!(
                "File(s) already exists in target directory: {}\nUse `--force` to overwrite.",
                existing.join(", ")
            );
        }

        Ok(())
    }

    pub fn entry(&mut self, path: impl Into<PathBuf>) -> &mut Vec<u8> {
        self.0.entry(path.into()).or_default()
    }

    pub fn write(&self, target_dir: &Path, force: bool) -> Result<()> {
        // Check that the paths do not exist and that we can write.
        if !force {
            self.check_exist(target_dir)?;
        }

        for (path, content) in &self.0 {
            let full_path = target_dir.join(path);
            write_to_file(&full_path, content)?;
        }

        Ok(())
    }

    pub fn extend(&mut self, other: FileSet) {
        self.0.extend(other.0);
    }

    pub fn into_names(self) -> Vec<PathBuf> {
        self.0.into_keys().collect()
    }
}

impl Default for FileSet {
    fn default() -> Self {
        FileSet::new()
    }
}

fn write_to_file(path: impl AsRef<Path>, data: &[u8]) -> Result<()> {
    let path = path.as_ref();

    let parent = path
        .parent()
        .ok_or_else(|| eyre!("Cannot get parent of `{}`", path.to_string_lossy()))?;
    std::fs::create_dir_all(parent)
        .wrap_err_with(|| format!("Cannot create directory `{}`", parent.to_string_lossy()))?;

    std::fs::write(path, data)
        .wrap_err_with(|| format!("Cannot create: {}", path.to_string_lossy()))
}
