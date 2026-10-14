pub(crate) mod cpp;
pub mod ops;
pub(crate) mod python;
pub mod recipe;
pub(crate) mod rewrite;
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
    run_pipeline_with(
        ws,
        recipe,
        recipe_dir,
        root,
        vendors,
        &mut |inv, summary| {
            println!("[line {:>3}] {:<19} {}", inv.line, inv.op, summary);
        },
    )
}

// Same pipeline, with each op's summary handed to a callback instead of
// stdout: a caller that is not a terminal (a wasm host, a test) has somewhere
// to put the log.
pub fn run_pipeline_with(
    ws: &mut Workspace,
    recipe: &[recipe::Invocation],
    recipe_dir: &Path,
    root: &Path,
    vendors: BTreeMap<String, PathBuf>,
    log: &mut dyn FnMut(&recipe::Invocation, &str),
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
        log(inv, &summary);
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
    let unresolved = unresolved_relative_imports(ws)?;
    if !unresolved.is_empty() {
        bail!(
            "verify: relative imports name modules that are not in the port (pruned, \
             moved, or new upstream):\n  {}",
            unresolved.join("\n  ")
        );
    }
    let mismatched = mismatched_declarations(ws)?;
    if !mismatched.is_empty() {
        bail!(
            "verify: functions declared under torch-ext no longer match their \
             definitions (upstream changed a signature; update the binding):\n  {}",
            mismatched.join("\n  ")
        );
    }
    Ok(facts)
}

// A binding written locally declares the upstream entry points it registers.
// When upstream changes one of their signatures the port still generates, and
// the kernel fails to link; so every function declared in a C-family file
// under torch-ext must have a definition in the port with the same parameter
// types.
fn mismatched_declarations(ws: &Workspace) -> Result<Vec<String>> {
    let mut declared = BTreeMap::new();
    for path in ws.glob_str("torch-ext/**")? {
        if !cpp::is_source(&path) {
            continue;
        }
        for sig in rewrite::function_signatures(ws.get_text(&path)?) {
            if !sig.definition {
                let types = sig.types();
                declared.entry(sig.name).or_insert_with(Vec::new).push((
                    path.clone(),
                    types,
                    sig.params,
                ));
            }
        }
    }
    if declared.is_empty() {
        return Ok(Vec::new());
    }
    let mut defined: BTreeMap<String, Vec<(String, Vec<String>, String)>> = BTreeMap::new();
    for path in ws.paths() {
        if !cpp::is_source(&path) {
            continue;
        }
        let text = ws.get_text(&path)?;
        if !declared.keys().any(|name| text.contains(name.as_str())) {
            continue;
        }
        for sig in rewrite::function_signatures(text) {
            if sig.definition && declared.contains_key(&sig.name) {
                let types = sig.types();
                defined
                    .entry(sig.name)
                    .or_default()
                    .push((path.clone(), types, sig.params));
            }
        }
    }
    let mut out = Vec::new();
    for (name, decls) in &declared {
        let defs = defined.get(name).map_or(&[][..], Vec::as_slice);
        for (path, types, params) in decls {
            if defs.iter().any(|(_, t, _)| t == types) {
                continue;
            }
            let mut entry = format!("{path}: {name}({params})");
            if defs.is_empty() {
                entry.push_str("\n    has no definition in the port");
            }
            let show = |t: Option<&String>| t.map_or("nothing".to_string(), |t| format!("`{t}`"));
            for (def_path, def_types, def_params) in defs {
                let at = types
                    .iter()
                    .zip(def_types)
                    .take_while(|(a, b)| a == b)
                    .count();
                entry.push_str(&format!(
                    "\n    defined in {def_path}: {name}({def_params})\n      parameter {}: \
                     declared {}, defined {}",
                    at + 1,
                    show(types.get(at)),
                    show(def_types.get(at))
                ));
            }
            out.push(entry);
        }
    }
    Ok(out)
}

// Relative imports must land on a module in the tree. Upstream adding an
// import of a module the recipe prunes otherwise only fails when the kernel is
// imported. `torch-ext/<package>/_ops` is generated by the build.
fn unresolved_relative_imports(ws: &Workspace) -> Result<Vec<String>> {
    let paths: std::collections::BTreeSet<String> = ws.paths().into_iter().collect();
    let is_package = |dir: &str| {
        let prefix = format!("{dir}/");
        paths
            .range(prefix.clone()..)
            .next()
            .is_some_and(|p| p.starts_with(&prefix))
    };
    let is_module = |m: &str| {
        paths.contains(&format!("{m}.py"))
            || is_package(m)
            || m.strip_prefix("torch-ext/")
                .and_then(|r| r.strip_suffix("/_ops"))
                .is_some_and(|pkg| !pkg.contains('/'))
    };
    let mut out = Vec::new();
    for path in &paths {
        if !is_python(path) {
            continue;
        }
        for import in python::relative_imports(path, ws.get_text(path)?)? {
            let mut base: Vec<&str> = path.split('/').collect();
            base.pop();
            let shown = format!(
                "{path}: from {}{} import {}",
                ".".repeat(import.dots),
                import.module.join("."),
                if import.names.is_empty() {
                    "*".to_string()
                } else {
                    import.names.join(", ")
                }
            );
            if import.dots - 1 > base.len() {
                out.push(shown);
                continue;
            }
            base.truncate(base.len() + 1 - import.dots);
            let mut module = base.join("/");
            for part in &import.module {
                if !module.is_empty() {
                    module.push('/');
                }
                module.push_str(part);
            }
            if !is_module(&module) {
                out.push(shown);
                continue;
            }
            // `from pkg import name`: a name that is no submodule must be an
            // attribute set by the package's __init__, which is not checked.
            if is_package(&module) && !paths.contains(&format!("{module}/__init__.py")) {
                let missing: Vec<&String> = import
                    .names
                    .iter()
                    .filter(|n| !is_module(&format!("{module}/{}", n.replace('.', "/"))))
                    .collect();
                if !missing.is_empty() {
                    out.push(shown);
                }
            }
        }
    }
    Ok(out)
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
