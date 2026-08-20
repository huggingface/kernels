pub mod ops;
pub(crate) mod python;
pub mod recipe;
#[cfg(test)]
mod tests;
pub mod workspace;

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use workspace::Workspace;

pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

pub(crate) fn is_python(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|e| e == "py")
}

pub fn run_pipeline(
    ws: &mut Workspace,
    recipe: &[recipe::Invocation],
    recipe_dir: &Path,
    root: &Path,
    vendors: BTreeMap<String, PathBuf>,
) -> Result<ops::Facts> {
    let inputs = ops::Inputs {
        root: root.to_path_buf(),
        vendors,
    };
    let mut facts = ops::Facts::default();
    // Build every op before applying any, so an argument typo or a bad glob
    // anywhere in the recipe fails before the first mutation.
    let mut built = Vec::with_capacity(recipe.len());
    for inv in recipe {
        let op = ops::build(inv, recipe_dir)
            .with_context(|| format!("recipe line {}: {}", inv.line, inv.op))?;
        built.push((inv, op));
    }
    for (inv, op) in built {
        let summary = op
            .apply(ws, &inputs, &mut facts)
            .with_context(|| format!("recipe line {}: {}", inv.line, inv.op))?;
        println!("[line {:>3}] {:<19} {}", inv.line, inv.op, summary);
    }
    let changes = ws.changes();
    for path in changes.added.iter().chain(&changes.modified) {
        if is_python(path) {
            python::check_parses(path, ws.get_text(path)?)?;
        }
    }
    // The Hub imports a kernel under a build-variant directory name, so an
    // absolute self-import resolves to nothing at run time.
    let mut offenders = Vec::new();
    for path in ws.glob_str("torch-ext/**")? {
        if !is_python(&path) {
            continue;
        }
        let rest = &path["torch-ext/".len()..];
        let Some((pkg, _)) = rest.split_once('/') else {
            continue;
        };
        for found in python::absolute_self_imports(&path, ws.get_text(&path)?, pkg)? {
            offenders.push(format!("{path}: {found}"));
        }
    }
    if !offenders.is_empty() {
        bail!(
            "verify: absolute in-package imports remain under torch-ext (they must \
             be relative):\n  {}",
            offenders.join("\n  ")
        );
    }
    Ok(facts)
}

// Every field is derived from the inputs, so re-running the port reproduces
// this file byte-for-byte.
pub fn provenance_json(
    recipe_text: &str,
    recipe_version: u64,
    sources: &[ops::SourceRecord],
    tree: &str,
) -> String {
    use sha2::{Digest, Sha256};

    #[derive(serde::Serialize)]
    struct Pin<'a> {
        repo: &'a str,
        commit: &'a str,
    }

    #[derive(serde::Serialize)]
    struct Provenance<'a> {
        format: u64,
        recipe: String,
        runner: String,
        sources: BTreeMap<&'a str, Pin<'a>>,
        tree: String,
    }

    let recipe = hex(&Sha256::digest(recipe_text.as_bytes()));
    let record = Provenance {
        format: recipe_version,
        recipe: format!("sha256:{recipe}"),
        runner: format!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")),
        sources: sources
            .iter()
            .map(|s| {
                (
                    s.name.as_str(),
                    Pin {
                        repo: &s.repo,
                        commit: &s.commit,
                    },
                )
            })
            .collect(),
        tree: format!("sha256:{tree}"),
    };
    let mut json = serde_json::to_string_pretty(&record).expect("provenance serializes");
    json.push('\n');
    json
}
