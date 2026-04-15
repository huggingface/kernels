use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use eyre::{Context, Result};

use crate::pyproject::FileSet;

const GITHUB_RAW_BASE_TEMPLATE: &str =
    "https://raw.githubusercontent.com/huggingface/kernels/main/kernel-builder/skills";

#[derive(Clone, Debug, ValueEnum)]
pub enum SkillId {
    CudaKernels,
    RocmKernels,
}

impl SkillId {
    fn as_str(&self) -> &'static str {
        match self {
            SkillId::CudaKernels => "cuda-kernels",
            SkillId::RocmKernels => "rocm-kernels",
        }
    }
}

impl Default for SkillId {
    fn default() -> Self {
        SkillId::CudaKernels
    }
}

struct Targets {
    codex: PathBuf,
    claude: PathBuf,
    opencode: PathBuf,
}

fn global_targets() -> Result<Targets> {
    let home = dirs::home_dir().ok_or_else(|| eyre::eyre!("Cannot determine home directory"))?;
    let config_dir =
        dirs::config_dir().ok_or_else(|| eyre::eyre!("Cannot determine config directory"))?;
    Ok(Targets {
        codex: home.join(".codex/skills"),
        claude: home.join(".claude/skills"),
        opencode: config_dir.join("opencode/skills"),
    })
}

fn local_targets() -> Targets {
    Targets {
        codex: PathBuf::from(".codex/skills"),
        claude: PathBuf::from(".claude/skills"),
        opencode: PathBuf::from(".opencode/skills"),
    }
}

fn github_raw_base(skill_id: &SkillId) -> String {
    format!("{GITHUB_RAW_BASE_TEMPLATE}/{}", skill_id.as_str())
}

fn download(url: &str) -> Result<String> {
    let body = reqwest::blocking::get(url)
        .wrap_err_with(|| format!("Failed to fetch {url}"))?
        .error_for_status()
        .wrap_err_with(|| format!("HTTP error for {url}"))?
        .text()
        .wrap_err("Failed to read response body")?;
    Ok(body)
}

fn download_manifest(skill_id: &SkillId) -> Result<Vec<String>> {
    let url = format!("{}/manifest.txt", github_raw_base(skill_id));
    let raw = download(&url)?;
    let entries: Vec<String> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    Ok(entries)
}

fn download_file(skill_id: &SkillId, rel_path: &str) -> Result<String> {
    let url = format!("{}/{rel_path}", github_raw_base(skill_id));
    download(&url)
}

fn remove_existing(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if path.is_symlink() || path.is_file() {
        fs::remove_file(path).wrap_err_with(|| format!("Cannot remove {}", path.display()))?;
    } else if path.is_dir() {
        fs::remove_dir_all(path)
            .wrap_err_with(|| format!("Cannot remove directory {}", path.display()))?;
    }
    Ok(())
}

fn install_to(target: impl AsRef<Path>, force: bool, skill_id: &SkillId) -> Result<PathBuf> {
    let target = target.as_ref();

    fs::create_dir_all(target)
        .wrap_err_with(|| format!("Cannot create directory {}", target.display()))?;

    let target = fs::canonicalize(target)
        .wrap_err_with(|| format!("Cannot canonicalize {}", target.display()))?;

    let dest = target.join(skill_id.as_str());

    if dest.exists() {
        if !force {
            eyre::bail!(
                "Skill already exists at {}.\nRe-run with --force to overwrite.",
                dest.display()
            );
        }
        remove_existing(&dest)?;
    }

    let manifest = download_manifest(skill_id)?;
    let mut fileset = FileSet::new();
    for rel_path in &manifest {
        let content = download_file(skill_id, rel_path)?;
        fileset
            .entry(rel_path)
            .extend_from_slice(content.as_bytes());
    }
    fileset.write(&dest, force)?;

    Ok(dest)
}

pub fn add_skill(
    skill_id: SkillId,
    claude: bool,
    codex: bool,
    opencode: bool,
    global: bool,
    dest: Option<PathBuf>,
    force: bool,
) -> Result<()> {
    if !claude && !codex && !opencode && dest.is_none() {
        eyre::bail!("Pick a destination via --claude, --codex, --opencode, or --dest.");
    }

    let mut install_targets: Vec<PathBuf> = Vec::new();

    if global {
        let targets = global_targets()?;
        if claude {
            install_targets.push(targets.claude);
        }
        if codex {
            install_targets.push(targets.codex);
        }
        if opencode {
            install_targets.push(targets.opencode);
        }
    } else {
        let targets = local_targets();
        if claude {
            install_targets.push(targets.claude);
        }
        if codex {
            install_targets.push(targets.codex);
        }
        if opencode {
            install_targets.push(targets.opencode);
        }
    }

    if let Some(d) = dest {
        install_targets.push(d);
    }

    for target in &install_targets {
        let installed_path = install_to(target, force, &skill_id)?;
        println!(
            "Installed '{}' to {}",
            skill_id.as_str(),
            installed_path.display()
        );
    }

    Ok(())
}
