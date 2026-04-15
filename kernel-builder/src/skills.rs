use std::fs;
use std::path::PathBuf;

use eyre::{Context, Result};

pub const DEFAULT_SKILL_ID: &str = "cuda-kernels";
pub const SUPPORTED_SKILL_IDS: &[&str] = &["cuda-kernels", "rocm-kernels"];

const GITHUB_RAW_BASE_TEMPLATE: &str =
    "https://raw.githubusercontent.com/huggingface/kernels/main/kernel-builder/skills";

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

fn github_raw_base(skill_id: &str) -> String {
    format!("{GITHUB_RAW_BASE_TEMPLATE}/{skill_id}")
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

fn download_manifest(skill_id: &str) -> Result<Vec<String>> {
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

fn download_file(skill_id: &str, rel_path: &str) -> Result<String> {
    let url = format!("{}/{rel_path}", github_raw_base(skill_id));
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

fn install_to(target: &PathBuf, force: bool, skill_id: &str) -> Result<PathBuf> {
    let target = fs::canonicalize(target).unwrap_or_else(|_| target.clone());

    fs::create_dir_all(&target)
        .wrap_err_with(|| format!("Cannot create directory {}", target.display()))?;

    let dest = target.join(skill_id);

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
    for rel_path in &manifest {
        let content = download_file(skill_id, rel_path)?;
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
    skill_id: &str,
    claude: bool,
    codex: bool,
    opencode: bool,
    global: bool,
    dest: Option<PathBuf>,
    force: bool,
) -> Result<()> {
    if !SUPPORTED_SKILL_IDS.contains(&skill_id) {
        let supported = SUPPORTED_SKILL_IDS.join(", ");
        eyre::bail!("Unsupported skill '{skill_id}'. Supported skills: {supported}");
    }

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
        let installed_path = install_to(target, force, skill_id)?;
        println!(
            "Installed '{}' to {}",
            skill_id,
            installed_path.display()
        );
    }

    Ok(())
}
