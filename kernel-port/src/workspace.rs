use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub struct Pattern {
    raw: String,
    matcher: globset::GlobMatcher,
}

impl std::str::FromStr for Pattern {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        // literal_separator keeps `*` from crossing a `/`; only `**` does.
        let matcher = globset::GlobBuilder::new(raw)
            .literal_separator(true)
            .build()
            .with_context(|| format!("invalid glob {raw:?}"))?
            .compile_matcher();
        Ok(Self {
            raw: raw.to_string(),
            matcher,
        })
    }
}

impl std::fmt::Debug for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.raw, f)
    }
}

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.raw)
    }
}

pub struct Workspace {
    files: BTreeMap<String, Vec<u8>>,
    // Load-time snapshot. The change set is the diff against it.
    initial: BTreeMap<String, Vec<u8>>,
}

pub struct ChangeSet {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
}

impl Workspace {
    pub fn load(root: &Path) -> Result<Self> {
        let mut files = BTreeMap::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
                .with_context(|| format!("reading {}", dir.display()))?
                .map(|e| Ok(e?.path()))
                .collect::<Result<_>>()?;
            entries.sort();
            for path in entries {
                let name = path.file_name().unwrap().to_string_lossy();
                if name == ".git" {
                    continue;
                }
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    let content = std::fs::read(&path)
                        .with_context(|| format!("reading {}", path.display()))?;
                    files.insert(rel, content);
                }
            }
        }
        Ok(Self {
            initial: files.clone(),
            files,
        })
    }

    pub fn initial_bytes(&self, path: &str) -> Option<&[u8]> {
        self.initial.get(path).map(std::vec::Vec::as_slice)
    }

    pub fn current_bytes(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(std::vec::Vec::as_slice)
    }

    pub fn tree_hash(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        // Path and length are hashed with the content so no two trees collide
        // by shifting bytes across a boundary.
        for (path, content) in &self.files {
            hasher.update(path.as_bytes());
            hasher.update([0]);
            hasher.update((content.len() as u64).to_le_bytes());
            hasher.update(content);
        }
        crate::hex(&hasher.finalize())
    }

    pub fn from_files(files: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            initial: files.clone(),
            files,
        }
    }

    pub fn paths(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }

    pub fn glob(&self, pattern: &Pattern) -> Vec<String> {
        self.files
            .keys()
            .filter(|p| pattern.matcher.is_match(p.as_str()))
            .cloned()
            .collect()
    }

    pub fn glob_str(&self, pattern: &str) -> Result<Vec<String>> {
        Ok(self.glob(&pattern.parse()?))
    }

    pub fn get_text(&self, path: &str) -> Result<&str> {
        let bytes = self
            .files
            .get(path)
            .with_context(|| format!("no such file in workspace: {path}"))?;
        std::str::from_utf8(bytes).with_context(|| format!("{path} is not valid UTF-8"))
    }

    pub fn set_text(&mut self, path: &str, content: String) {
        self.files.insert(path.to_string(), content.into_bytes());
    }

    pub fn insert(&mut self, path: &str, content: Vec<u8>) {
        self.files.insert(path.to_string(), content);
    }

    pub fn delete(&mut self, path: &str) -> Result<()> {
        if self.files.remove(path).is_none() {
            bail!("no such file in workspace: {path}");
        }
        Ok(())
    }

    pub fn rename(&mut self, from: &str, to: &str) -> Result<Vec<(String, String)>> {
        let mut pairs = Vec::new();
        if self.files.contains_key(from) {
            pairs.push((from.to_string(), to.to_string()));
        } else {
            let prefix = format!("{from}/");
            for p in self.files.keys() {
                if let Some(rest) = p.strip_prefix(&prefix) {
                    pairs.push((p.clone(), format!("{to}/{rest}")));
                }
            }
            if pairs.is_empty() {
                bail!("{from:?} matches no file or directory in the workspace");
            }
        }
        for (_, new) in &pairs {
            if self.files.contains_key(new) {
                bail!("destination {new:?} already exists");
            }
        }
        for (old, new) in &pairs {
            let content = self.files.remove(old).unwrap();
            self.files.insert(new.clone(), content);
        }
        Ok(pairs)
    }

    pub fn changes(&self) -> ChangeSet {
        let mut added = Vec::new();
        let mut modified = Vec::new();
        let mut deleted = Vec::new();
        for (path, content) in &self.files {
            match self.initial.get(path) {
                None => added.push(path.clone()),
                Some(old) if old != content => modified.push(path.clone()),
                Some(_) => {}
            }
        }
        for path in self.initial.keys() {
            if !self.files.contains_key(path) {
                deleted.push(path.clone());
            }
        }
        ChangeSet {
            added,
            modified,
            deleted,
        }
    }

    // Wiped first, so the output is a function of the recipe alone and never a
    // merge with a previous run.
    pub fn materialize_into(&self, out: &Path) -> Result<()> {
        if out.exists() {
            std::fs::remove_dir_all(out)
                .with_context(|| format!("clearing previous output {}", out.display()))?;
        }
        for (path, content) in &self.files {
            let disk = out.join(path);
            if let Some(parent) = disk.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&disk, content)
                .with_context(|| format!("writing {}", disk.display()))?;
        }
        Ok(())
    }

    pub fn materialize(&self, root: &Path) -> Result<()> {
        let changes = self.changes();
        for path in changes.added.iter().chain(&changes.modified) {
            let disk = root.join(path);
            if let Some(parent) = disk.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&disk, &self.files[path])
                .with_context(|| format!("writing {}", disk.display()))?;
        }
        for path in &changes.deleted {
            let disk = root.join(path);
            std::fs::remove_file(&disk).with_context(|| format!("removing {}", disk.display()))?;
            // remove_dir only succeeds on an empty directory, which is the
            // stop condition for pruning what the delete emptied.
            let mut dir = disk.parent().map(Path::to_path_buf);
            while let Some(d) = dir {
                if d == root || std::fs::remove_dir(&d).is_err() {
                    break;
                }
                dir = d.parent().map(Path::to_path_buf);
            }
        }
        Ok(())
    }
}
