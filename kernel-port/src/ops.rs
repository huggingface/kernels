use crate::python::{self, DottedPath};
use crate::recipe::{Args, Invocation};
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

#[derive(Debug)]
pub struct CommitSha(String);

impl CommitSha {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for CommitSha {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() != 40 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("commit must be a full 40-character hex SHA, got {s:?}");
        }
        Ok(Self(s.to_string()))
    }
}

impl std::fmt::Display for CommitSha {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .context("running git")?;
    if !out.status.success() {
        bail!(
            "git {} failed in {}: {}",
            args.join(" "),
            root.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

// So a pin written as .../repo.git matches a checkout cloned from .../repo.
fn normalize_git_url(url: &str) -> String {
    url.trim_end_matches('/')
        .trim_end_matches(".git")
        .to_string()
}

fn check_clean_checkout(root: &Path, what: &str) -> Result<()> {
    let status = git(root, &["status", "--porcelain"])?;
    if !status.is_empty() {
        bail!(
            "{what} checkout {} is not clean (modified or untracked files present); \
             the output would not be reproducible from the pin:\n{status}",
            root.display()
        );
    }
    Ok(())
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
    EnsureInit(EnsureInit),
}

pub fn build(inv: &Invocation, recipe_dir: &Path) -> Result<Op> {
    let mut args = inv.take_args();
    let op = match inv.op.as_str() {
        "source" => Op::Source(Source {
            repo: args.take("repo")?,
            commit: args.take("commit")?.parse()?,
        }),
        "vendor" => Op::Vendor(Vendor {
            name: args.take("name")?,
            repo: args.take("repo")?,
            commit: args.take("commit")?.parse()?,
            path: args.take("path")?.trim_matches('/').to_string(),
            to: args.take("to")?.trim_matches('/').to_string(),
        }),
        "delete" => Op::Delete(Delete {
            pattern: args.take("in")?.parse()?,
        }),
        "move" => Op::Move(Move {
            from: args.take("from")?,
            to: args.take("to")?,
        }),
        "replace" => Op::Replace(Replace::build(&mut args)?),
        "strip_suffix" => Op::StripSuffix(StripSuffix::build(&mut args)?),
        "expect" => Op::Expect(Expect::build(&mut args)?),
        "overlay" => Op::Overlay(Overlay {
            source: recipe_dir.join(args.take("from")?),
        }),
        "prune" => Op::Prune(Prune::build(&mut args)?),
        "kernel" => Op::Kernel(Kernel::build(&mut args)?),
        "manifest" => Op::Manifest(Manifest::build(&mut args)?),
        "relativize_imports" => Op::RelativizeImports(RelativizeImports {
            pattern: args.take("in")?.parse()?,
            package_root: args.take("package_root")?.trim_end_matches('/').to_string(),
            changes: args.take_usize_opt("changes")?,
            root_relative: match args.take_opt("root_relative").as_deref() {
                None | Some("false") => false,
                Some("true") => true,
                Some(other) => bail!("root_relative must be true or false, got {other:?}"),
            },
        }),
        "remap_module" => Op::RemapModule(RemapModule {
            pattern: args.take("in")?.parse()?,
            from: args.take("from")?.parse()?,
            to: args.take("to")?.parse()?,
            changes: args.take_usize_opt("changes")?,
        }),
        "convert_import" => Op::ConvertImport(ConvertImport {
            pattern: args.take("in")?.parse()?,
            prefix: args.take("prefix")?.parse()?,
            changes: args.take_usize_opt("changes")?,
        }),
        "ensure_init" => Op::EnsureInit(EnsureInit {
            under: args.take("under")?.trim_matches('/').to_string(),
            changes: args.take_usize_opt("changes")?,
        }),
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
            Self::EnsureInit(op) => op.apply(ws),
        }
    }
}

#[derive(Debug)]
pub struct Source {
    repo: String,
    commit: CommitSha,
}

impl Source {
    fn apply(&self, inputs: &Inputs, facts: &mut Facts) -> Result<String> {
        let head = git(&inputs.root, &["rev-parse", "HEAD"]).with_context(|| {
            format!(
                "source pinning requires --dir to be a git checkout of {}",
                self.repo
            )
        })?;
        if head != self.commit.as_str() {
            bail!(
                "checkout is at {head} but this port is written against {} - \
                 re-clone at the pinned commit, or re-verify the port and update the pin",
                self.commit
            );
        }
        // A checkout with no origin still passes; only a mismatch fails.
        if let Ok(origin) = git(&inputs.root, &["remote", "get-url", "origin"])
            && normalize_git_url(&origin) != normalize_git_url(&self.repo)
        {
            bail!(
                "checkout origin is {origin} but this port is for {} - wrong repository",
                self.repo
            );
        }
        check_clean_checkout(&inputs.root, "source")?;
        facts.record_source("upstream", &self.repo, self.commit.as_str());
        Ok(format!("verified {} @ {}", self.repo, &head[..12]))
    }
}

#[derive(Debug)]
pub struct Delete {
    pattern: Pattern,
}

impl Delete {
    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let matches = ws.glob(&self.pattern);
        if matches.is_empty() {
            bail!("{:?} matches nothing", self.pattern);
        }
        let n = matches.len();
        for path in matches {
            ws.delete(&path)?;
        }
        Ok(format!("removed {n} file(s)"))
    }
}

#[derive(Debug)]
pub struct Move {
    from: String,
    to: String,
}

impl Move {
    fn apply(&self, ws: &mut Workspace, facts: &mut Facts) -> Result<String> {
        let pairs = ws.rename(&self.from, &self.to)?;
        let n = pairs.len();
        facts.moved.extend(pairs);
        Ok(format!("moved {n} file(s) to {:?}", self.to))
    }
}

#[derive(Debug)]
pub struct Replace {
    pattern: Pattern,
    find: String,
    with: String,
    count: usize,
}

impl Replace {
    fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            find: args.take("find")?,
            with: args.take("with")?,
            count: args.take_usize("count")?,
        };
        if op.find.is_empty() {
            bail!("find must not be empty");
        }
        Ok(op)
    }

    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let files = ws.glob(&self.pattern);
        if files.is_empty() {
            bail!("{:?} matches no files", self.pattern);
        }
        let mut found = 0;
        let mut per_file = Vec::new();
        for path in &files {
            let n = ws.get_text(path)?.matches(&self.find).count();
            if n > 0 {
                per_file.push((path.clone(), n));
            }
            found += n;
        }
        if found != self.count {
            bail!(
                "expected exactly {} match(es) of {:?} in {:?}, found {} ({})",
                self.count,
                self.find,
                self.pattern,
                found,
                if per_file.is_empty() {
                    "no files matched the text".to_string()
                } else {
                    per_file
                        .iter()
                        .map(|(p, n)| format!("{p}: {n}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            );
        }
        let n_files = per_file.len();
        for (path, _) in per_file {
            let updated = ws.get_text(&path)?.replace(&self.find, &self.with);
            ws.set_text(&path, updated);
        }
        Ok(format!("{found} replacement(s) in {n_files} file(s)"))
    }
}

#[derive(Debug)]
pub struct StripSuffix {
    pattern: Pattern,
    suffix: String,
    files: usize,
}

impl StripSuffix {
    fn build(args: &mut Args) -> Result<Self> {
        let op = Self {
            pattern: args.take("in")?.parse()?,
            suffix: args.take("suffix")?,
            files: args.take_usize("files")?,
        };
        if op.suffix.is_empty() {
            bail!("suffix must not be empty");
        }
        Ok(op)
    }

    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let paths = ws.glob(&self.pattern);
        if paths.is_empty() {
            bail!("{:?} matches no files", self.pattern);
        }
        if paths.len() != self.files {
            bail!(
                "expected exactly {} file(s) for {:?}, found {}",
                self.files,
                self.pattern,
                paths.len()
            );
        }
        // Every file is checked before any is written, so a partial strip
        // cannot land.
        let mut updates = Vec::with_capacity(paths.len());
        for path in paths {
            let text = ws.get_text(&path)?;
            let Some(stripped) = text.strip_suffix(&self.suffix) else {
                bail!("{path:?} does not end with pinned suffix {:?}", self.suffix);
            };
            updates.push((path, stripped.to_string()));
        }
        for (path, updated) in updates {
            ws.set_text(&path, updated);
        }
        Ok(format!("stripped suffix from {} file(s)", self.files))
    }
}

#[derive(Debug)]
pub struct Expect {
    pattern: Pattern,
    kind: ExpectKind,
}

#[derive(Debug)]
enum ExpectKind {
    Text { find: String, count: usize },
    FileCount(usize),
}

impl Expect {
    fn build(args: &mut Args) -> Result<Self> {
        let pattern = args.take("in")?.parse()?;
        let kind = match (args.take_opt("find"), args.take_usize_opt("files")?) {
            (Some(find), None) => {
                if find.is_empty() {
                    bail!("find must not be empty");
                }
                ExpectKind::Text {
                    find,
                    count: args.take_usize("count")?,
                }
            }
            (None, Some(n)) => ExpectKind::FileCount(n),
            (Some(_), Some(_)) => bail!("find and files are mutually exclusive"),
            (None, None) => bail!("expect requires either find=.../count=N or files=N"),
        };
        Ok(Self { pattern, kind })
    }

    fn apply(&self, ws: &Workspace) -> Result<String> {
        let files = ws.glob(&self.pattern);
        match &self.kind {
            ExpectKind::FileCount(expected) => {
                if files.len() != *expected {
                    bail!(
                        "expected {:?} to match exactly {expected} file(s), found {} - \
                         the upstream file set drifted; update the port definition ({})",
                        self.pattern,
                        files.len(),
                        files.join(", ")
                    );
                }
                Ok(format!("{expected} file(s), as expected"))
            }
            ExpectKind::Text { find, count } => {
                if files.is_empty() {
                    bail!("{:?} matches no files", self.pattern);
                }
                let mut found = 0;
                for path in &files {
                    found += ws.get_text(path)?.matches(find).count();
                }
                if found != *count {
                    bail!(
                        "expected exactly {count} occurrence(s) of {find:?} in {:?}, found \
                         {found} - upstream changed text this port depends on; update the \
                         port definition",
                        self.pattern
                    );
                }
                Ok(format!("{found} occurrence(s), as expected"))
            }
        }
    }
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

#[derive(Debug)]
pub struct Overlay {
    source: PathBuf,
}

impl Overlay {
    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        if !self.source.is_dir() {
            bail!("{} is not a directory", self.source.display());
        }
        let copied = copy_tree(ws, &self.source, std::string::ToString::to_string, false)?;
        if copied == 0 {
            bail!("{} contains no files", self.source.display());
        }
        Ok(format!("copied {copied} file(s)"))
    }
}

#[derive(Debug)]
pub struct Vendor {
    name: String,
    repo: String,
    commit: CommitSha,
    path: String,
    to: String,
}

impl Vendor {
    fn apply(&self, ws: &mut Workspace, inputs: &Inputs, facts: &mut Facts) -> Result<String> {
        let Some(checkout) = inputs.vendors.get(&self.name) else {
            bail!(
                "no checkout supplied for vendor {:?}; pass --vendor {}=<dir> on the CLI",
                self.name,
                self.name
            );
        };
        let head = git(checkout, &["rev-parse", "HEAD"]).with_context(|| {
            format!(
                "vendor {:?} must be a git checkout of {}",
                self.name, self.repo
            )
        })?;
        if head != self.commit.as_str() {
            bail!(
                "vendor {:?} checkout is at {head} but this port is written against {}",
                self.name,
                self.commit
            );
        }
        if let Ok(origin) = git(checkout, &["remote", "get-url", "origin"])
            && normalize_git_url(&origin) != normalize_git_url(&self.repo)
        {
            bail!(
                "vendor {:?} checkout origin is {origin} but this port expects {}",
                self.name,
                self.repo
            );
        }
        check_clean_checkout(checkout, "vendor")?;
        facts.record_source(&self.name, &self.repo, self.commit.as_str());

        let src_root = checkout.join(&self.path);
        if !src_root.is_dir() {
            bail!(
                "{} is not a directory in the vendor checkout",
                src_root.display()
            );
        }
        let copied = copy_tree(ws, &src_root, |rel| format!("{}/{rel}", self.to), true)?;
        if copied == 0 {
            bail!("{} contains no files", src_root.display());
        }
        Ok(format!(
            "vendored {copied} file(s) from {}:{} @ {}",
            self.name,
            self.path,
            &head[..12]
        ))
    }
}

#[derive(Debug)]
pub struct Prune {
    keep: Vec<Pattern>,
}

impl Prune {
    fn build(args: &mut Args) -> Result<Self> {
        let keep = glob_comma_list(&args.take("keep")?)
            .iter()
            .map(|s| s.parse())
            .collect::<Result<Vec<Pattern>>>()?;
        if keep.is_empty() {
            bail!("keep must list at least one glob");
        }
        Ok(Self { keep })
    }

    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let mut kept = std::collections::BTreeSet::new();
        for glob in &self.keep {
            let matches = ws.glob(glob);
            if matches.is_empty() {
                bail!("keep glob {glob:?} matches nothing");
            }
            kept.extend(matches);
        }
        let doomed: Vec<String> = ws
            .glob_str("**")?
            .into_iter()
            .filter(|p| !kept.contains(p))
            .collect();
        let n = doomed.len();
        for path in doomed {
            ws.delete(&path)?;
        }
        Ok(format!("removed {n} file(s), kept {}", kept.len()))
    }
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
pub struct Kernel {
    name: String,
    backend: String,
    src: Vec<Pattern>,
    include: Vec<String>,
    depends: Vec<String>,
    capabilities: Vec<String>,
    cxx_flags: Vec<String>,
    cuda_flags: Vec<String>,
    cuda_minver: Option<String>,
    rocm_archs: Vec<String>,
    repeat_src: Vec<String>,
}

impl Kernel {
    fn build(args: &mut Args) -> Result<Self> {
        Ok(Self {
            name: args.take("name")?,
            backend: args.take("backend")?,
            src: glob_list(&args.take("src")?, "src")?,
            include: comma_list(&args.take_opt("include").unwrap_or_default()),
            depends: comma_list(&args.take_opt("depends").unwrap_or_else(|| "torch".into())),
            capabilities: comma_list(&args.take_opt("capabilities").unwrap_or_default()),
            cxx_flags: comma_list(&args.take_opt("cxx_flags").unwrap_or_default()),
            cuda_flags: comma_list(&args.take_opt("cuda_flags").unwrap_or_default()),
            cuda_minver: args.take_opt("cuda_minver"),
            rocm_archs: comma_list(&args.take_opt("rocm_archs").unwrap_or_default()),
            repeat_src: comma_list(&args.take_opt("repeat_src").unwrap_or_default()),
        })
    }

    fn apply(&self, ws: &Workspace, facts: &mut Facts) -> Result<String> {
        if facts.kernels.iter().any(|k| k.name == self.name) {
            bail!("kernel section {:?} already declared", self.name);
        }
        let mut src = glob_union(ws, &self.src, "src")?;
        for path in &self.repeat_src {
            if !src.contains(path) {
                bail!("repeat_src path {path:?} is not selected by src globs");
            }
            src.push(path.clone());
        }
        src.sort();
        for dir in &self.include {
            check_include_dir(ws, dir)?;
        }
        let n = src.len();
        facts.kernels.push(KernelSection {
            name: self.name.clone(),
            backend: self.backend.clone(),
            cxx_flags: self.cxx_flags.clone(),
            cuda_flags: self.cuda_flags.clone(),
            cuda_minver: self.cuda_minver.clone(),
            rocm_archs: self.rocm_archs.clone(),
            depends: self.depends.clone(),
            include: self.include.clone(),
            capabilities: self.capabilities.clone(),
            src,
        });
        Ok(format!(
            "declared [kernel.{}] with {n} src file(s)",
            self.name
        ))
    }
}

// build.toml is generated, never overlaid. A field this op cannot emit is a
// reason to extend the op.
#[derive(Debug)]
pub struct Manifest {
    name: String,
    version: Option<String>,
    license: Option<String>,
    edition: Option<String>,
    upstream: Option<String>,
    backends: Vec<String>,
    repo_id: Option<String>,
    hub_branch: Option<String>,
    python_depends: Vec<String>,
    cuda_minver: Option<String>,
    cuda_maxver: Option<String>,
    cuda_python_depends: Vec<String>,
    kind: ManifestKind,
}

#[derive(Debug)]
enum ManifestKind {
    Noarch {
        pyext: Vec<String>,
    },
    Torch {
        torch_src: Vec<Pattern>,
        pyext: Vec<String>,
        include: Vec<String>,
        stable_abi_version: Option<String>,
        stable_abi: Vec<(String, String)>,
    },
}

impl Manifest {
    fn build(args: &mut Args) -> Result<Self> {
        let backends = comma_list(&args.take("backends")?);
        if backends.is_empty() {
            bail!("backends must list at least one backend");
        }
        let version = args.take_opt("version");
        if let Some(v) = &version {
            v.parse::<u64>()
                .with_context(|| format!("version must be an integer, got {v:?}"))?;
        }
        let edition = args.take_opt("edition");
        if let Some(e) = &edition {
            e.parse::<u64>()
                .with_context(|| format!("edition must be an integer, got {e:?}"))?;
        }
        let noarch = match args.take_opt("noarch").as_deref() {
            None => false,
            Some("true") => true,
            Some(other) => bail!("noarch must be true if given, got {other:?}"),
        };
        let kind = if noarch {
            if args.take_opt("torch_src").is_some() {
                bail!("torch_src does not apply when noarch=true");
            }
            if args.take_opt("stable_abi").is_some() {
                bail!("stable_abi does not apply when noarch=true");
            }
            if args.take_opt("stable_abi_version").is_some() {
                bail!("stable_abi_version does not apply when noarch=true");
            }
            if args.take_opt("torch_include").is_some() {
                bail!("torch_include does not apply when noarch=true");
            }
            ManifestKind::Noarch {
                pyext: comma_list(&args.take_opt("noarch_pyext").unwrap_or_default()),
            }
        } else {
            if args.take_opt("noarch_pyext").is_some() {
                bail!("noarch_pyext requires noarch=true");
            }
            let stable_abi_version = args.take_opt("stable_abi_version");
            let pyext = comma_list(&args.take_opt("torch_pyext").unwrap_or_default());
            let stable_abi = comma_list(&args.take_opt("stable_abi").unwrap_or_default())
                .iter()
                .map(|pair| {
                    pair.split_once('=')
                        .map(|(b, v)| (b.to_string(), v.to_string()))
                        .with_context(|| {
                            format!("stable_abi entry {pair:?} must be backend=version")
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            if stable_abi_version.is_some() && !stable_abi.is_empty() {
                bail!("stable_abi_version and stable_abi are mutually exclusive");
            }
            ManifestKind::Torch {
                torch_src: glob_list(&args.take("torch_src")?, "torch_src")?,
                pyext,
                include: comma_list(&args.take_opt("torch_include").unwrap_or_default()),
                stable_abi_version,
                stable_abi,
            }
        };
        Ok(Self {
            name: args.take("name")?,
            version,
            license: args.take_opt("license"),
            edition,
            upstream: args.take_opt("upstream"),
            backends,
            repo_id: args.take_opt("repo_id"),
            hub_branch: args.take_opt("hub_branch"),
            python_depends: comma_list(&args.take_opt("python_depends").unwrap_or_default()),
            cuda_minver: args.take_opt("cuda_minver"),
            cuda_maxver: args.take_opt("cuda_maxver"),
            cuda_python_depends: comma_list(
                &args.take_opt("cuda_python_depends").unwrap_or_default(),
            ),
            kind,
        })
    }

    fn toml_list(key: &str, items: &[String]) -> String {
        if items.len() == 1 {
            format!("{key} = [{}]\n", toml_str_list(items))
        } else {
            let mut block = format!("{key} = [\n");
            for item in items {
                block.push_str(&format!("    {item:?},\n"));
            }
            block.push_str("]\n");
            block
        }
    }

    fn src_block(files: &[String]) -> String {
        let mut block = String::from("src = [\n");
        for f in files {
            block.push_str(&format!("    {f:?},\n"));
        }
        block.push(']');
        block
    }

    fn general_section(&self) -> String {
        let mut toml = String::from("[general]\n");
        toml.push_str(&format!("name = {:?}\n", self.name));
        if let Some(version) = &self.version {
            toml.push_str(&format!("version = {version}\n"));
        }
        if let Some(license) = &self.license {
            toml.push_str(&format!("license = {license:?}\n"));
        }
        if let Some(edition) = &self.edition {
            toml.push_str(&format!("edition = {edition}\n"));
        }
        if let Some(upstream) = &self.upstream {
            toml.push_str(&format!("upstream = {upstream:?}\n"));
        }
        toml.push_str(&Self::toml_list("backends", &self.backends));
        if !self.python_depends.is_empty() {
            toml.push_str(&Self::toml_list("python-depends", &self.python_depends));
        }
        if self.cuda_minver.is_some()
            || self.cuda_maxver.is_some()
            || !self.cuda_python_depends.is_empty()
        {
            toml.push_str("\n[general.cuda]\n");
            if let Some(minver) = &self.cuda_minver {
                toml.push_str(&format!("minver = {minver:?}\n"));
            }
            if let Some(maxver) = &self.cuda_maxver {
                toml.push_str(&format!("maxver = {maxver:?}\n"));
            }
            if !self.cuda_python_depends.is_empty() {
                toml.push_str(&Self::toml_list(
                    "python-depends",
                    &self.cuda_python_depends,
                ));
            }
        }
        if self.repo_id.is_some() || self.hub_branch.is_some() {
            toml.push_str("\n[general.hub]\n");
            if let Some(repo_id) = &self.repo_id {
                toml.push_str(&format!("repo-id = {repo_id:?}\n"));
            }
            if let Some(branch) = &self.hub_branch {
                toml.push_str(&format!("branch = {branch:?}\n"));
            }
        }
        toml
    }

    fn apply(&self, ws: &mut Workspace, facts: &Facts) -> Result<String> {
        let mut toml = self.general_section();

        let summary = match &self.kind {
            ManifestKind::Noarch { pyext } => {
                if !facts.kernels.is_empty() {
                    bail!("noarch=true but kernel sections were declared");
                }
                toml.push_str("\n[torch-noarch]\n");
                if !pyext.is_empty() {
                    toml.push_str(&format!("pyext = [{}]\n", toml_str_list(pyext)));
                }
                toml.push_str("\n[kernel]\n");
                "wrote build.toml (noarch)".to_string()
            }
            ManifestKind::Torch {
                torch_src,
                pyext,
                include,
                stable_abi_version,
                stable_abi,
            } => {
                let torch_files = glob_union(ws, torch_src, "torch_src")?;
                if facts.kernels.is_empty() {
                    bail!("no kernel sections declared; add `kernel` statements before manifest");
                }
                toml.push_str("\n[torch]\n");
                if let Some(version) = stable_abi_version {
                    toml.push_str(&format!("stable-abi = {version:?}\n"));
                }
                if !pyext.is_empty() {
                    toml.push_str(&Self::toml_list("pyext", pyext));
                }
                for dir in include {
                    check_include_dir(ws, dir)?;
                }
                if !include.is_empty() {
                    toml.push_str(&format!("include = [{}]\n", toml_str_list(include)));
                }
                toml.push_str(&Self::src_block(&torch_files));
                toml.push('\n');
                if !stable_abi.is_empty() {
                    toml.push_str("\n[torch.stable-abi]\n");
                    for (backend, version) in stable_abi {
                        toml.push_str(&format!("{backend} = {version:?}\n"));
                    }
                }
                for k in &facts.kernels {
                    toml.push_str(&format!("\n[kernel.{}]\n", k.name));
                    toml.push_str(&format!("backend = {:?}\n", k.backend));
                    if !k.cxx_flags.is_empty() {
                        toml.push_str("cxx-flags = [\n");
                        for f in &k.cxx_flags {
                            toml.push_str(&format!("    {f:?},\n"));
                        }
                        toml.push_str("]\n");
                    }
                    if !k.capabilities.is_empty() {
                        toml.push_str(&format!(
                            "cuda-capabilities = [{}]\n",
                            toml_str_list(&k.capabilities)
                        ));
                    }
                    if !k.cuda_flags.is_empty() {
                        toml.push_str("cuda-flags = [\n");
                        for f in &k.cuda_flags {
                            toml.push_str(&format!("    {f:?},\n"));
                        }
                        toml.push_str("]\n");
                    }
                    if let Some(minver) = &k.cuda_minver {
                        toml.push_str(&format!("cuda-minver = {minver:?}\n"));
                    }
                    toml.push_str(&format!("depends = [{}]\n", toml_str_list(&k.depends)));
                    if !k.rocm_archs.is_empty() {
                        toml.push_str(&Self::toml_list("rocm-archs", &k.rocm_archs));
                    }
                    if !k.include.is_empty() {
                        toml.push_str(&format!("include = [{}]\n", toml_str_list(&k.include)));
                    }
                    toml.push_str(&Self::src_block(&k.src));
                    toml.push('\n');
                }
                format!(
                    "wrote build.toml ({} torch src, {} kernel section(s))",
                    torch_files.len(),
                    facts.kernels.len()
                )
            }
        };

        ws.set_text("build.toml", toml);
        Ok(summary)
    }
}

#[derive(Debug)]
pub struct ConvertImport {
    pattern: Pattern,
    prefix: DottedPath,
    changes: Option<usize>,
}

impl ConvertImport {
    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "converted", |path, src| {
            python::convert_imports_source(path, src, &self.prefix)
        })
    }
}

#[derive(Debug)]
pub struct EnsureInit {
    under: String,
    changes: Option<usize>,
}

impl EnsureInit {
    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let files = ws.glob_str(&format!("{}/**", self.under))?;
        if files.is_empty() {
            bail!("no files under {:?}", self.under);
        }
        let mut dirs = std::collections::BTreeSet::new();
        dirs.insert(self.under.clone());
        for path in &files {
            if !crate::is_python(path) {
                continue;
            }
            let mut dir = path.rsplit_once('/').map(|(d, _)| d.to_string());
            while let Some(d) = dir {
                if d.len() <= self.under.len() {
                    break;
                }
                dirs.insert(d.clone());
                dir = d.rsplit_once('/').map(|(p, _)| p.to_string());
            }
        }
        let mut added = 0;
        for dir in dirs {
            let init = format!("{dir}/__init__.py");
            if ws.current_bytes(&init).is_none() {
                ws.insert(&init, Vec::new());
                added += 1;
            }
        }
        check_changes_pin(self.changes, added)?;
        Ok(format!("added {added} missing __init__.py file(s)"))
    }
}

#[derive(Debug)]
pub struct RemapModule {
    pattern: Pattern,
    from: DottedPath,
    to: DottedPath,
    changes: Option<usize>,
}

impl RemapModule {
    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "rewrote", |path, src| {
            python::remap_source(path, src, &self.from, &self.to)
        })
    }
}

#[derive(Debug)]
pub struct RelativizeImports {
    pattern: Pattern,
    package_root: String,
    changes: Option<usize>,
    root_relative: bool,
}

impl RelativizeImports {
    fn package_of(&self, path: &str) -> Result<Vec<String>> {
        let parent = self
            .package_root
            .rfind('/')
            .map_or("", |i| &self.package_root[..=i]);
        let Some(rel) = path.strip_prefix(parent) else {
            bail!("{path:?} is not under package_root {:?}", self.package_root);
        };
        let mut components: Vec<String> = rel.split('/').map(str::to_string).collect();
        components.pop();
        if components.is_empty() {
            bail!("{path:?} sits at the package root's parent; it has no package");
        }
        Ok(components)
    }

    fn apply(&self, ws: &mut Workspace) -> Result<String> {
        apply_rewrite(ws, &self.pattern, self.changes, "rewrote", |path, src| {
            let package = self.package_of(path)?;
            if self.root_relative {
                python::relativize_source_from_root(path, src, &package)
            } else {
                python::relativize_source(path, src, &package)
            }
        })
    }
}
