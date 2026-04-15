use std::fs;
use std::path::PathBuf;

use eyre::{Context, Result};

const DEFAULT_SKILL_ID: &str = "cuda-kernels";
const GITHUB_RAW_BASE: &str =
    "https://raw.githubusercontent.com/huggingface/kernels/main/kernel-builder/skills/cuda-kernels";
const MANIFEST_URL: &str = concat!(
    "https://raw.githubusercontent.com/huggingface/kernels/main/kernel-builder/skills/cuda-kernels",
    "/manifest.txt"
);

struct Targets {
    codex: PathBuf,
    claude: PathBuf,
    opencode: PathBuf,
}

fn global_targets() -> Result<Targets> {
    let home = dirs::home_dir().ok_or_else(|| eyre::eyre!("Cannot determine home directory"))?;
    Ok(Targets {
        codex: home.join(".codex/skills"),
        claude: home.join(".claude/skills"),
        opencode: home.join(".config/opencode/skills"),
    })
}

fn local_targets() -> Targets {
    Targets {
        codex: PathBuf::from(".codex/skills"),
        claude: PathBuf::from(".claude/skills"),
        opencode: PathBuf::from(".opencode/skills"),
    }
}

fn download(url: &str) -> Result<String> {
    let body = ureq::get(url)
        .call()
        .wrap_err_with(|| format!("Failed to fetch {url}"))?
        .into_body()
        .read_to_string()
        .wrap_err("Failed to read response body")?;
    Ok(body)
}

fn download_manifest() -> Result<Vec<String>> {
    let raw = download(MANIFEST_URL)?;
    let entries: Vec<String> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    Ok(entries)
}

fn download_file(rel_path: &str) -> Result<String> {
    let url = format!("{GITHUB_RAW_BASE}/{rel_path}");
    download(&url)
}

fn remove_existing(path: &PathBuf) -> Result<()> {
    if path.is_symlink() || path.is_file() {
        fs::remove_file(path)
            .wrap_err_with(|| format!("Cannot remove {}", path.display()))?;
    } else if path.is_dir() {
        fs::remove_dir_all(path)
            .wrap_err_with(|| format!("Cannot remove directory {}", path.display()))?;
    }
    Ok(())
}

fn install_to(target: &PathBuf, force: bool) -> Result<PathBuf> {
    let target = fs::canonicalize(target).unwrap_or_else(|_| target.clone());

    fs::create_dir_all(&target)
        .wrap_err_with(|| format!("Cannot create directory {}", target.display()))?;

    let dest = target.join(DEFAULT_SKILL_ID);

    if dest.exists() {
        if !force {
            eyre::bail!(
                "Skill already exists at {}.\nRe-run with --force to overwrite.",
                dest.display()
            );
        }
        remove_existing(&dest)?;
    }

    let manifest = download_manifest()?;
    for rel_path in &manifest {
        let content = download_file(rel_path)?;
        let output_file = dest.join(rel_path);
        if let Some(parent) = output_file.parent() {
            fs::create_dir_all(parent)
                .wrap_err_with(|| format!("Cannot create directory {}", parent.display()))?;
        }
        fs::write(&output_file, &content)
            .wrap_err_with(|| format!("Cannot write {}", output_file.display()))?;
    }

    Ok(dest)
}

pub fn add_skill(
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
        let installed_path = install_to(target, force)?;
        println!(
            "Installed '{}' to {}",
            DEFAULT_SKILL_ID,
            installed_path.display()
        );
    }

    Ok(())
}
