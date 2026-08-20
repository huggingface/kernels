use anyhow::{Context, Result, bail};
use clap::Parser;
use kernel_port::{provenance_json, recipe, run_pipeline, workspace::Workspace};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "kernel-port")]
struct Cli {
    #[arg(required_unless_present = "recipe")]
    recipe_path: Option<PathBuf>,
    #[arg(
        short = 'e',
        long = "recipe",
        value_name = "TEXT",
        conflicts_with = "recipe_path"
    )]
    recipe: Option<String>,
    #[arg(long, required_unless_present = "files")]
    dir: Option<PathBuf>,
    #[arg(long = "file", value_name = "PATH=CONTENT", conflicts_with = "dir")]
    files: Vec<String>,
    #[arg(long)]
    print: bool,
    #[arg(long, conflicts_with = "dry_run")]
    out: Option<PathBuf>,
    #[arg(long)]
    dry_run: bool,
    #[arg(long)]
    diff: bool,
    #[arg(long = "vendor", value_name = "NAME=DIR")]
    vendor: Vec<String>,
    #[arg(long, requires = "out")]
    partial: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let Cli {
        recipe_path,
        recipe: recipe_inline,
        dir,
        files,
        print,
        out,
        dry_run,
        diff,
        vendor,
        partial,
    } = Cli::parse();

    let mut vendors = std::collections::BTreeMap::new();
    for spec in &vendor {
        let (name, path) = spec
            .split_once('=')
            .with_context(|| format!("--vendor {spec:?} must be name=dir"))?;
        vendors.insert(name.to_string(), PathBuf::from(path));
    }

    let (text, recipe_dir) = match (&recipe_path, recipe_inline) {
        (Some(path), None) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let dir = path
                .canonicalize()
                .with_context(|| format!("resolving {}", path.display()))?
                .parent()
                .map(PathBuf::from)
                .context("recipe has no parent directory")?;
            (text, dir)
        }
        // An inline recipe has no file to be relative to, so paths it names
        // (`overlay from=`) resolve against the working directory.
        (None, Some(text)) => (text, PathBuf::from(".")),
        _ => unreachable!("clap requires exactly one of <RECIPE_PATH> and --recipe"),
    };
    let recipe = recipe::parse(&text)?;
    check_declares_version(&recipe, recipe_path.as_deref())?;

    if let Some(dir) = &dir
        && !dir.is_dir()
    {
        bail!("--dir {} is not a directory", dir.display());
    }
    if let (Some(out), Some(dir)) = (&out, &dir) {
        check_out_disjoint(out, dir)?;
    }
    // Both modes converge on one in-memory workspace: --dir reads it off disk,
    // --file builds it from the command line.
    let (mut ws, root) = match &dir {
        Some(dir) => (Workspace::load(dir)?, dir.clone()),
        None => (inline_workspace(&files)?, PathBuf::from(".")),
    };
    let facts = match run_pipeline(&mut ws, &recipe.ops, &recipe_dir, &root, vendors) {
        Ok(facts) => facts,
        Err(err) => {
            if partial {
                let out = out.as_ref().unwrap();
                ws.materialize_into(out)?;
                eprintln!(
                    "--partial: workspace state before the failing op written to {}",
                    out.display()
                );
            }
            return Err(err);
        }
    };

    let changes = ws.changes();
    for path in &changes.added {
        println!("A {path}");
    }
    for path in &changes.modified {
        println!("M {path}");
    }
    for path in &changes.deleted {
        println!("D {path}");
    }
    if changes.added.is_empty() && changes.modified.is_empty() && changes.deleted.is_empty() {
        println!("no changes");
    }
    if diff {
        print_diffs(&ws, &changes.modified, &facts.moved);
    }

    // An inline workspace has nowhere to be written back to, so printing the
    // result is the whole point of the run.
    if print || (dir.is_none() && out.is_none()) {
        print_tree(&ws);
    }
    match (&out, &dir, dry_run) {
        (Some(out), _, _) => {
            ws.materialize_into(out)?;
            let provenance = provenance_json(
                &text,
                recipe.effective_version(),
                &facts.sources,
                &ws.tree_hash(),
            );
            std::fs::write(out.join(".port-provenance.json"), provenance)?;
        }
        (None, Some(dir), false) => ws.materialize(dir)?,
        (None, _, _) => {}
    }
    Ok(())
}

// A recipe kept on disk is re-run later, by someone else, against a newer
// build, so it has to say which format it was written against. An inline
// recipe is gone the moment the command finishes.
fn check_declares_version(parsed: &recipe::Recipe, path: Option<&Path>) -> Result<()> {
    if let Some(path) = path
        && parsed.version.is_none()
    {
        bail!(
            "{}: recipe does not declare its format; add `recipe version={}` as \
             the first line",
            path.display(),
            recipe::VERSION
        );
    }
    Ok(())
}

// --out is wiped and regenerated on every run, so it must not overlap the
// source. The directory need not exist yet; its parent must.
fn check_out_disjoint(out: &Path, dir: &Path) -> Result<()> {
    let dir_canon = dir.canonicalize()?;
    let out_canon = match out.canonicalize() {
        Ok(p) => p,
        Err(_) => out
            .parent()
            .filter(|p| p.as_os_str().is_empty() || p.exists())
            .map(|p| {
                let base = if p.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    p.to_path_buf()
                };
                Ok::<_, anyhow::Error>(base.canonicalize()?.join(out.file_name().unwrap()))
            })
            .transpose()?
            .with_context(|| format!("--out {}: parent directory does not exist", out.display()))?,
    };
    if out_canon.starts_with(&dir_canon) || dir_canon.starts_with(&out_canon) {
        bail!(
            "--out {} overlaps --dir {}; the output directory is wiped on every \
             run and must be disjoint from the source",
            out.display(),
            dir.display()
        );
    }
    Ok(())
}

fn inline_workspace(files: &[String]) -> Result<Workspace> {
    let mut map = std::collections::BTreeMap::new();
    for spec in files {
        let (path, content) = spec
            .split_once('=')
            .with_context(|| format!("--file {spec:?} must be path=content"))?;
        if map
            .insert(path.to_string(), content.as_bytes().to_vec())
            .is_some()
        {
            bail!("--file {path:?} given more than once");
        }
    }
    Ok(Workspace::from_files(map))
}

fn print_tree(ws: &Workspace) {
    for path in ws.paths() {
        println!("\nFILE: {path}");
        match std::str::from_utf8(ws.current_bytes(&path).unwrap_or_default()) {
            Ok(text) => print!("{text}{}", if text.ends_with('\n') { "" } else { "\n" }),
            Err(_) => println!("<binary>"),
        }
    }
}

fn print_diffs(ws: &Workspace, modified: &[String], moved: &[(String, String)]) {
    let mut pairs: Vec<(&str, &str)> = modified.iter().map(|p| (p.as_str(), p.as_str())).collect();
    // A moved file is diffed old path against new, rather than showing up as a
    // delete plus an add.
    for (old, new) in moved {
        let changed = match (ws.initial_bytes(old), ws.current_bytes(new)) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        };
        if changed {
            pairs.push((old.as_str(), new.as_str()));
        }
    }
    pairs.sort_by_key(|&(_, new)| new.to_string());
    for (old_path, new_path) in pairs {
        let old = ws.initial_bytes(old_path).unwrap_or_default();
        let new = ws.current_bytes(new_path).unwrap_or_default();
        match (std::str::from_utf8(old), std::str::from_utf8(new)) {
            (Ok(old), Ok(new)) => {
                let diff = similar::TextDiff::from_lines(old, new);
                print!(
                    "--- a/{old_path}\n+++ b/{new_path}\n{}",
                    diff.unified_diff().context_radius(2)
                );
            }
            _ => println!("--- {new_path}: binary file changed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::inline_workspace;

    #[test]
    fn inline_files_split_on_the_first_equals() {
        let ws = inline_workspace(&["a.py=x = 1 == 1\n".to_string()]).unwrap();
        assert_eq!(ws.get_text("a.py").unwrap(), "x = 1 == 1\n");
    }

    #[test]
    fn inline_files_reject_bad_specs() {
        let err = |specs: &[String]| inline_workspace(specs).err().unwrap().to_string();
        assert!(err(&["a.py".to_string()]).contains("must be path=content"));
        let dup = ["a.py=1".to_string(), "a.py=2".to_string()];
        assert!(err(&dup).contains("more than once"));
    }
}
