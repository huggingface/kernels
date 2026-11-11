// Nix archive (NAR) hash of a source tree, the `hash` that `fetchgit` checks
// for a checkout at the same commit. See section 5.2 of Eelco Dolstra's thesis
// for the format.

use anyhow::{Context, Result};
use base64::Engine;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The SRI hash of `root`. A git checkout is hashed as its tracked files, since
/// that is what `fetchgit` checks out. Any other tree is hashed whole.
pub fn hash(root: &Path, git: bool) -> Result<String> {
    let tracked = if git { Some(tracked(root)?) } else { None };
    let mut nar = Sha256::new();
    str(&mut nar, b"nix-archive-1");
    node(&mut nar, root, Path::new(""), tracked.as_ref())?;
    let digest = base64::engine::general_purpose::STANDARD.encode(nar.finalize());
    Ok(format!("sha256-{digest}"))
}

// Tracked files and the directories that contain them. A submodule is kept as
// an empty directory, as `fetchgit` without `fetchSubmodules` leaves it.
fn tracked(root: &Path) -> Result<BTreeSet<PathBuf>> {
    let out = super::git::git(root, &["ls-files", "-z"])?;
    Ok(out
        .split('\0')
        .filter(|e| !e.is_empty())
        .flat_map(|path| Path::new(path).ancestors().map(Path::to_path_buf))
        .collect())
}

fn node(
    nar: &mut Sha256,
    path: &Path,
    rel: &Path,
    tracked: Option<&BTreeSet<PathBuf>>,
) -> Result<()> {
    let meta = fs::symlink_metadata(path).with_context(|| format!("reading {}", path.display()))?;
    str(nar, b"(");
    str(nar, b"type");
    if meta.is_symlink() {
        str(nar, b"symlink");
        str(nar, b"target");
        str(nar, fs::read_link(path)?.as_os_str().as_encoded_bytes());
    } else if meta.is_dir() {
        str(nar, b"directory");
        let mut entries = fs::read_dir(path)?
            .map(|e| e.map(|e| e.file_name()))
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
        for name in entries {
            let child = rel.join(&name);
            if (rel.as_os_str().is_empty() && name == ".git")
                || tracked.is_some_and(|t| !t.contains(&child))
            {
                continue;
            }
            str(nar, b"entry");
            str(nar, b"(");
            str(nar, b"name");
            str(nar, name.as_encoded_bytes());
            str(nar, b"node");
            node(nar, &path.join(&name), &child, tracked)?;
            str(nar, b")");
        }
    } else {
        str(nar, b"regular");
        if executable(&meta) {
            str(nar, b"executable");
            str(nar, b"");
        }
        str(nar, b"contents");
        str(nar, &fs::read(path)?);
    }
    str(nar, b")");
    Ok(())
}

// A length-prefixed string, padded to a multiple of 8 bytes.
fn str(nar: &mut Sha256, bytes: &[u8]) {
    nar.update((bytes.len() as u64).to_le_bytes());
    nar.update(bytes);
    nar.update(&[0; 8][..(8 - bytes.len() % 8) % 8]);
}

#[cfg(unix)]
fn executable(meta: &fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o100 != 0
}

#[cfg(not(unix))]
fn executable(_: &fs::Metadata) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    // `fetchgit` checks out tracked files only, so ignored files must not count.
    #[test]
    fn checkout_hashes_tracked_files() {
        let temp = tempfile::tempdir().unwrap();
        let (repo, copy) = (temp.path().join("repo"), temp.path().join("copy"));
        for dir in [&repo, &copy] {
            fs::create_dir_all(dir.join("pkg")).unwrap();
            fs::write(dir.join(".gitignore"), "*.pyc\n").unwrap();
            fs::write(dir.join("pkg/a.py"), "a = 1\n").unwrap();
        }
        fs::write(repo.join("pkg/a.pyc"), "ignored").unwrap();
        let git = |args: &[&str]| {
            assert!(
                Command::new("git")
                    .current_dir(&repo)
                    .args(args)
                    .status()
                    .unwrap()
                    .success()
            );
        };
        git(&["init", "-q"]);
        git(&["add", "."]);
        assert_eq!(
            super::hash(&repo, true).unwrap(),
            super::hash(&copy, false).unwrap()
        );
    }
}
