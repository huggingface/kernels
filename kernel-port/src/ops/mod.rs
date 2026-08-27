mod convert_import;
mod delete;
mod ensure_import;
mod ensure_init;
mod expect;
mod git;
mod kernel;
mod manifest;
mod r#move;
mod overlay;
mod prune;
mod relativize_imports;
mod remap_module;
mod replace;
mod source;
mod strip_suffix;
mod vendor;

pub use convert_import::ConvertImport;
pub use delete::Delete;
pub use ensure_import::EnsureImport;
pub use ensure_init::EnsureInit;
pub use expect::Expect;
pub use kernel::Kernel;
pub use manifest::Manifest;
pub use r#move::Move;
pub use overlay::Overlay;
pub use prune::Prune;
pub use relativize_imports::RelativizeImports;
pub use remap_module::RemapModule;
pub use replace::Replace;
pub use source::Source;
pub use strip_suffix::StripSuffix;
pub use vendor::Vendor;

use crate::recipe::Invocation;
use crate::workspace::{Pattern, Workspace};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct Inputs {
    pub root: PathBuf,
    pub vendors: std::collections::BTreeMap<String, PathBuf>,
}

#[derive(Default)]
pub struct Facts {
    pub moved: Vec<(String, String)>,
    pub kernels: Vec<KernelSection>,
    pub sources: Vec<SourceRecord>,
}

pub struct SourceRecord {
    pub name: String,
    pub repo: String,
    pub commit: String,
}

impl Facts {
    fn record_source(&mut self, name: &str, repo: &str, commit: &str) {
        if !self.sources.iter().any(|s| s.name == name) {
            self.sources.push(SourceRecord {
                name: name.to_string(),
                repo: repo.to_string(),
                commit: commit.to_string(),
            });
        }
    }
}

pub struct KernelSection {
    pub name: String,
    pub backend: String,
    pub cxx_flags: Vec<String>,
    pub cuda_flags: Vec<String>,
    pub cuda_minver: Option<String>,
    pub rocm_archs: Vec<String>,
    pub depends: Vec<String>,
    pub include: Vec<String>,
    pub capabilities: Vec<String>,
    pub src: Vec<String>,
}

fn check_changes_pin(expected: Option<usize>, actual: usize) -> Result<()> {
    if let Some(expected) = expected
        && actual != expected
    {
        bail!(
            "expected exactly {expected} change(s) but made {actual} - upstream \
                 drifted; review the new rewrites and update changes="
        );
    }
    Ok(())
}

// Comma-separated, where `,,` is an escaped comma: compiler flags are values
// that contain commas.
fn comma_list(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = value.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ',' {
            if chars.peek() == Some(&',') {
                chars.next();
                cur.push(',');
            } else {
                let item = cur.trim();
                if !item.is_empty() {
                    out.push(item.to_string());
                }
                cur.clear();
            }
        } else {
            cur.push(c);
        }
    }
    let item = cur.trim();
    if !item.is_empty() {
        out.push(item.to_string());
    }
    out
}

fn toml_str_list(items: &[String]) -> String {
    items
        .iter()
        .map(|s| format!("{s:?}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn apply_rewrite(
    ws: &mut Workspace,
    pattern: &Pattern,
    changes: Option<usize>,
    verb: &str,
    mut rewrite: impl FnMut(&str, &str) -> Result<Option<(String, usize)>>,
) -> Result<String> {
    let files = ws.glob(pattern);
    if files.is_empty() {
        bail!("{pattern:?} matches no files");
    }
    let mut n_files = 0;
    let mut n_imports = 0;
    for path in files {
        if !crate::is_python(&path) {
            continue;
        }
        if let Some((updated, count)) = rewrite(&path, ws.get_text(&path)?)? {
            ws.set_text(&path, updated);
            n_files += 1;
            n_imports += count;
        }
    }
    // The pin counts rewritten imports, not files, so a new upstream file
    // cannot be rewritten without moving it.
    check_changes_pin(changes, n_imports)?;
    Ok(format!("{verb} {n_imports} import(s) in {n_files} file(s)"))
}

fn copy_tree(
    ws: &mut Workspace,
    source: &Path,
    dest: impl Fn(&str) -> String,
    refuse_existing: bool,
) -> Result<usize> {
    let mut stack = vec![source.to_path_buf()];
    let mut copied = 0usize;
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .map(|e| Ok(e?.path()))
            .collect::<Result<_>>()?;
        entries.sort();
        for path in entries {
            if path.file_name().is_some_and(|n| n == ".git") {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(source)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let dest = dest(&rel);
                if refuse_existing && ws.current_bytes(&dest).is_some() {
                    bail!("vendor destination {dest:?} already exists in the workspace");
                }
                let content =
                    std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                ws.insert(&dest, content);
                copied += 1;
            }
        }
    }
    Ok(copied)
}

// Commas inside a `{a,b}` alternation belong to the glob, not to the list.
fn glob_comma_list(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for c in value.chars() {
        match c {
            '{' => {
                depth += 1;
                cur.push(c);
            }
            '}' => {
                depth = depth.saturating_sub(1);
                cur.push(c);
            }
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out.iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn glob_list(value: &str, what: &str) -> Result<Vec<Pattern>> {
    let globs = glob_comma_list(value)
        .iter()
        .map(|s| s.parse())
        .collect::<Result<Vec<Pattern>>>()?;
    if globs.is_empty() {
        bail!("{what} must list at least one glob");
    }
    Ok(globs)
}

fn glob_union(ws: &Workspace, globs: &[Pattern], what: &str) -> Result<Vec<String>> {
    let mut union = std::collections::BTreeSet::new();
    for glob in globs {
        let matches = ws.glob(glob);
        if matches.is_empty() {
            bail!("{what} glob {glob:?} matches no files");
        }
        union.extend(matches);
    }
    Ok(union.into_iter().collect())
}

// An include directory is named, not globbed, so a typo would otherwise
// surface at build time rather than here.
fn check_include_dir(ws: &Workspace, dir: &str) -> Result<()> {
    let glob = if dir.trim_matches('/') == "." {
        "**".to_string()
    } else {
        format!("{}/**", dir.trim_matches('/'))
    };
    if ws.glob_str(&glob)?.is_empty() {
        bail!("include directory {dir:?} contains no files");
    }
    Ok(())
}

#[derive(Debug)]
pub enum Op {
    Source(Source),
    Vendor(Vendor),
    Delete(Delete),
    Move(Move),
    Replace(Replace),
    StripSuffix(StripSuffix),
    Expect(Expect),
    Overlay(Overlay),
    Prune(Prune),
    Kernel(Kernel),
    Manifest(Manifest),
    RelativizeImports(RelativizeImports),
    RemapModule(RemapModule),
    ConvertImport(ConvertImport),
    EnsureImport(EnsureImport),
    EnsureInit(EnsureInit),
}

pub fn build(inv: &Invocation, recipe_dir: &Path) -> Result<Op> {
    let mut args = inv.take_args();
    let op = match inv.op.as_str() {
        "source" => Op::Source(Source::build(&mut args)?),
        "vendor" => Op::Vendor(Vendor::build(&mut args)?),
        "delete" => Op::Delete(Delete::build(&mut args)?),
        "move" => Op::Move(Move::build(&mut args)?),
        "replace" => Op::Replace(Replace::build(&mut args)?),
        "strip_suffix" => Op::StripSuffix(StripSuffix::build(&mut args)?),
        "expect" => Op::Expect(Expect::build(&mut args)?),
        "overlay" => Op::Overlay(Overlay::build(&mut args, recipe_dir)?),
        "prune" => Op::Prune(Prune::build(&mut args)?),
        "kernel" => Op::Kernel(Kernel::build(&mut args)?),
        "manifest" => Op::Manifest(Manifest::build(&mut args)?),
        "relativize_imports" => Op::RelativizeImports(RelativizeImports::build(&mut args)?),
        "remap_module" => Op::RemapModule(RemapModule::build(&mut args)?),
        "convert_import" => Op::ConvertImport(ConvertImport::build(&mut args)?),
        "ensure_import" => Op::EnsureImport(EnsureImport::build(&mut args)?),
        "ensure_init" => Op::EnsureInit(EnsureInit::build(&mut args)?),
        other => bail!("unknown op {other:?}"),
    };
    args.finish()?;
    Ok(op)
}

impl Op {
    pub fn apply(&self, ws: &mut Workspace, inputs: &Inputs, facts: &mut Facts) -> Result<String> {
        match self {
            Self::Source(op) => op.apply(inputs, facts),
            Self::Vendor(op) => op.apply(ws, inputs, facts),
            Self::Delete(op) => op.apply(ws),
            Self::Move(op) => op.apply(ws, facts),
            Self::Replace(op) => op.apply(ws),
            Self::StripSuffix(op) => op.apply(ws),
            Self::Expect(op) => op.apply(ws),
            Self::Overlay(op) => op.apply(ws),
            Self::Prune(op) => op.apply(ws),
            Self::Kernel(op) => op.apply(ws, facts),
            Self::Manifest(op) => op.apply(ws, facts),
            Self::RelativizeImports(op) => op.apply(ws),
            Self::RemapModule(op) => op.apply(ws),
            Self::ConvertImport(op) => op.apply(ws),
            Self::EnsureImport(op) => op.apply(ws),
            Self::EnsureInit(op) => op.apply(ws),
        }
    }
}
