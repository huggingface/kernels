// Python edits go through libcst, so comments, quoting and layout outside the
// rewritten import survive byte-for-byte.
use anyhow::{Context, Result, bail};
use libcst_native::{
    AssignTargetExpression, Codegen, CodegenState, CompoundStatement, Expression, Import,
    ImportAlias, ImportFrom, ImportNames, Module, NameOrAttribute, OrElse, SmallStatement,
    Statement, Suite,
};

#[derive(Debug)]
pub struct DottedPath(Vec<String>);

impl DottedPath {
    pub fn parts(&self) -> &[String] {
        &self.0
    }
}

impl std::str::FromStr for DottedPath {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<String> = s.split('.').map(str::to_string).collect();
        if parts.iter().any(String::is_empty) {
            bail!("{s:?} must be a non-empty dotted module path");
        }
        Ok(Self(parts))
    }
}

// A parsed module borrows the text it came from, so caching one means keeping
// the two together. `self_cell` owns the source alongside the tree that points
// into it, which is what lets a `Module` outlive the caller's `&str`.
self_cell::self_cell!(
    struct OwnedModule {
        owner: String,
        #[covariant]
        dependent: Module,
    }
);

// Parsing is what a port spends its time on, and roughly half of it re-reads
// bytes that were already parsed: an op matches a file an earlier op left
// untouched, and the end-of-pipeline verify re-reads the whole tree once more.
// Both checks below are pure functions of the text, so their answers are
// memoized on a content digest instead of being recomputed.
//
// The digest is SHA-256, not a fast hash: a collision here would silently skip
// a validation, and hashing is a rounding error next to a libcst parse.
mod memo {
    use super::OwnedModule;
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::rc::Rc;

    pub type Digest = [u8; 32];

    // The wasm playground re-runs the pipeline on every recipe edit against one
    // long-lived module, so the tables are capped rather than left to grow.
    const CAP: usize = 8192;

    // Parsed modules are held by content, so an op that matches a file an
    // earlier op left alone reuses the tree instead of rebuilding it. Sharing
    // is by `Rc` and the map borrow is released before the tree is handed out,
    // so a caller that parses another file while holding one cannot deadlock.
    thread_local! {
        static PARSES: RefCell<HashSet<Digest>> = RefCell::new(HashSet::new());
        static SELF_IMPORTS: RefCell<HashMap<(Digest, String), Vec<String>>> =
            RefCell::new(HashMap::new());
        static MODULES: RefCell<(HashMap<Digest, Rc<OwnedModule>>, usize)> =
            RefCell::new((HashMap::new(), 0));
    }

    // Syntax trees are far larger than the text they came from, so the module
    // cache is bounded by the source bytes it is holding rather than by entry
    // count. Ports run against trees far below this; the bound is there for the
    // playground, which keeps one module alive across many edits.
    const MODULE_BUDGET: usize = 8 << 20;

    pub fn module(key: &Digest) -> Option<Rc<OwnedModule>> {
        MODULES.with_borrow(|(map, _)| map.get(key).cloned())
    }

    pub fn note_module(key: Digest, owned: OwnedModule) -> Rc<OwnedModule> {
        let owned = Rc::new(owned);
        MODULES.with_borrow_mut(|(map, bytes)| {
            if *bytes >= MODULE_BUDGET {
                map.clear();
                *bytes = 0;
            }
            *bytes += owned.borrow_owner().len();
            map.insert(key, Rc::clone(&owned));
        });
        owned
    }

    pub fn digest(src: &str) -> Digest {
        use sha2::{Digest as _, Sha256};
        Sha256::digest(src.as_bytes()).into()
    }

    pub fn parses(key: Digest) -> bool {
        PARSES.with_borrow(|set| set.contains(&key))
    }

    pub fn note_parses(key: Digest) {
        PARSES.with_borrow_mut(|set| {
            if set.len() >= CAP {
                set.clear();
            }
            set.insert(key);
        });
    }

    pub fn self_imports(key: &(Digest, String)) -> Option<Vec<String>> {
        SELF_IMPORTS.with_borrow(|map| map.get(key).cloned())
    }

    pub fn note_self_imports(key: (Digest, String), found: Vec<String>) {
        SELF_IMPORTS.with_borrow_mut(|map| {
            if map.len() >= CAP {
                map.clear();
            }
            map.insert(key, found);
        });
    }
}

fn render<'a>(node: &impl Codegen<'a>) -> String {
    render_indented(node, &[])
}

fn render_indented<'a>(node: &impl Codegen<'a>, indents: &[&'a str]) -> String {
    let mut state = CodegenState {
        default_newline: "\n",
        default_indent: "    ",
        indent_tokens: indents.to_vec(),
        ..Default::default()
    };
    node.codegen(&mut state);
    state.tokens
}

fn check_roundtrip(module: &Module, src: &str) -> Result<()> {
    let mut state = CodegenState {
        default_newline: module.default_newline,
        default_indent: module.default_indent,
        ..Default::default()
    };
    module.codegen(&mut state);
    if state.tokens != src {
        bail!("libcst does not round-trip this file byte-for-byte; refusing to rewrite it");
    }
    Ok(())
}

fn flatten<'a>(node: &NameOrAttribute<'a>) -> Option<Vec<&'a str>> {
    match node {
        NameOrAttribute::N(name) => Some(vec![name.value]),
        NameOrAttribute::A(attr) => {
            let mut parts = flatten_expr(&attr.value)?;
            parts.push(attr.attr.value);
            Some(parts)
        }
    }
}

fn flatten_expr<'a>(expr: &Expression<'a>) -> Option<Vec<&'a str>> {
    match expr {
        Expression::Name(name) => Some(vec![name.value]),
        Expression::Attribute(attr) => {
            let mut parts = flatten_expr(&attr.value)?;
            parts.push(attr.attr.value);
            Some(parts)
        }
        _ => None,
    }
}

#[derive(Default)]
struct Imports<'m, 'a> {
    from_imports: Vec<(&'m ImportFrom<'a>, Vec<&'a str>)>,
    plain_imports: Vec<(&'m Import<'a>, Vec<&'a str>)>,
}

fn collect_small<'m, 'a>(
    small: &'m SmallStatement<'a>,
    stack: &[&'a str],
    out: &mut Imports<'m, 'a>,
) {
    match small {
        SmallStatement::ImportFrom(import) => out.from_imports.push((import, stack.to_vec())),
        SmallStatement::Import(import) => out.plain_imports.push((import, stack.to_vec())),
        _ => {}
    }
}

fn collect_imports<'m, 'a>(
    stmts: &'m [Statement<'a>],
    stack: &mut Vec<&'a str>,
    out: &mut Imports<'m, 'a>,
) {
    for stmt in stmts {
        match stmt {
            Statement::Simple(line) => {
                for small in &line.body {
                    collect_small(small, stack, out);
                }
            }
            Statement::Compound(compound) => collect_compound(compound, stack, out),
        }
    }
}

fn collect_compound<'m, 'a>(
    stmt: &'m CompoundStatement<'a>,
    stack: &mut Vec<&'a str>,
    out: &mut Imports<'m, 'a>,
) {
    match stmt {
        CompoundStatement::FunctionDef(f) => collect_suite(&f.body, stack, out),
        CompoundStatement::ClassDef(c) => collect_suite(&c.body, stack, out),
        CompoundStatement::If(i) => collect_if(i, stack, out),
        CompoundStatement::For(f) => {
            collect_suite(&f.body, stack, out);
            if let Some(e) = &f.orelse {
                collect_suite(&e.body, stack, out);
            }
        }
        CompoundStatement::While(w) => {
            collect_suite(&w.body, stack, out);
            if let Some(e) = &w.orelse {
                collect_suite(&e.body, stack, out);
            }
        }
        CompoundStatement::Try(t) => {
            collect_suite(&t.body, stack, out);
            for handler in &t.handlers {
                collect_suite(&handler.body, stack, out);
            }
            if let Some(e) = &t.orelse {
                collect_suite(&e.body, stack, out);
            }
            if let Some(f) = &t.finalbody {
                collect_suite(&f.body, stack, out);
            }
        }
        CompoundStatement::TryStar(t) => {
            collect_suite(&t.body, stack, out);
            for handler in &t.handlers {
                collect_suite(&handler.body, stack, out);
            }
            if let Some(e) = &t.orelse {
                collect_suite(&e.body, stack, out);
            }
            if let Some(f) = &t.finalbody {
                collect_suite(&f.body, stack, out);
            }
        }
        CompoundStatement::With(w) => collect_suite(&w.body, stack, out),
        CompoundStatement::Match(m) => {
            for case in &m.cases {
                collect_suite(&case.body, stack, out);
            }
        }
    }
}

fn collect_if<'m, 'a>(
    node: &'m libcst_native::If<'a>,
    stack: &mut Vec<&'a str>,
    out: &mut Imports<'m, 'a>,
) {
    collect_suite(&node.body, stack, out);
    if let Some(orelse) = &node.orelse {
        match orelse.as_ref() {
            OrElse::Elif(elif) => collect_if(elif, stack, out),
            OrElse::Else(e) => collect_suite(&e.body, stack, out),
        }
    }
}

fn collect_suite<'m, 'a>(
    suite: &'m Suite<'a>,
    stack: &mut Vec<&'a str>,
    out: &mut Imports<'m, 'a>,
) {
    match suite {
        Suite::IndentedBlock(block) => {
            stack.push(block.indent.unwrap_or("    "));
            collect_imports(&block.body, stack, out);
            stack.pop();
        }
        Suite::SimpleStatementSuite(line) => {
            for small in &line.body {
                collect_small(small, stack, out);
            }
        }
    }
}

fn module_imports<'m, 'a>(module: &'m Module<'a>) -> Imports<'m, 'a> {
    let mut imports = Imports::default();
    collect_imports(&module.body, &mut Vec::new(), &mut imports);
    imports
}

fn render_with_module<'a>(
    import: &ImportFrom<'a>,
    dots: usize,
    module: &str,
    indents: &[&'a str],
) -> String {
    let mut text = String::from("from");
    text.push_str(&render_indented(&import.whitespace_after_from, indents));
    text.push_str(&".".repeat(dots));
    text.push_str(module);
    text.push_str(&render_indented(&import.whitespace_before_import, indents));
    text.push_str("import");
    text.push_str(&render_indented(&import.whitespace_after_import, indents));
    if let Some(lpar) = &import.lpar {
        text.push_str(&render_indented(lpar, indents));
    }
    text.push_str(&render_indented(&import.names, indents));
    if let Some(rpar) = &import.rpar {
        text.push_str(&render_indented(rpar, indents));
    }
    if let Some(semi) = &import.semicolon {
        text.push_str(&render_indented(semi, indents));
    }
    text
}

struct Rewrite {
    old: String,
    new: String,
    nodes: usize,
}

fn add_rewrite(rewrites: &mut Vec<Rewrite>, old: String, new: String) {
    match rewrites.iter_mut().find(|r| r.old == old) {
        Some(existing) => {
            debug_assert_eq!(existing.new, new);
            existing.nodes += 1;
        }
        None => rewrites.push(Rewrite { old, new, nodes: 1 }),
    }
}

// Every parse in this module goes through here, including the ones that only
// want to know whether the text is still valid Python: a validated tree is
// worth keeping, because the op that reads that file next, and the verify pass
// at the end of the run, would otherwise rebuild it from scratch.
//
// The error is returned as libcst wrote it so that each caller can keep the
// wording of its own failure.
fn cached_module(src: &str) -> std::result::Result<std::rc::Rc<OwnedModule>, String> {
    let key = memo::digest(src);
    if let Some(hit) = memo::module(&key) {
        return Ok(hit);
    }
    let owned = OwnedModule::try_new(src.to_string(), |owner| {
        libcst_native::parse_module(owner.as_str(), None).map_err(|e| e.to_string())
    })?;
    memo::note_parses(key);
    Ok(memo::note_module(key, owned))
}

fn module_of(path: &str, src: &str) -> Result<std::rc::Rc<OwnedModule>> {
    cached_module(src).map_err(|e| anyhow::anyhow!("parsing {path}: {e}"))
}

// The round-trip check gates rewriting, not reading, so it stays outside the
// cache: `absolute_self_imports` only inspects imports and must not reject a
// file merely because libcst would reformat it.
fn parsed_module(path: &str, src: &str) -> Result<std::rc::Rc<OwnedModule>> {
    let owned = module_of(path, src)?;
    check_roundtrip(owned.borrow_dependent(), src).with_context(|| path.to_string())?;
    Ok(owned)
}

pub fn validate_ensure_import(from: &str, name: &str) -> Result<()> {
    let statement = format!("from {from} import {name}\n");
    let module = libcst_native::parse_module(&statement, None)
        .map_err(|e| anyhow::anyhow!("invalid import `from {from} import {name}`: {e}"))?;
    let [Statement::Simple(line)] = module.body.as_slice() else {
        bail!("invalid import `from {from} import {name}`");
    };
    let [SmallStatement::ImportFrom(import)] = line.body.as_slice() else {
        bail!("invalid import `from {from} import {name}`");
    };
    let ImportNames::Aliases(names) = &import.names else {
        bail!("import name must be a Python identifier, got {name:?}");
    };
    if names.len() != 1
        || names[0].asname.is_some()
        || flatten(&names[0].name).is_none_or(|parts| parts.as_slice() != [name])
    {
        bail!("import name must be a Python identifier, got {name:?}");
    }
    Ok(())
}

fn same_import_source(import: &ImportFrom<'_>, wanted: &ImportFrom<'_>) -> bool {
    import.relative.len() == wanted.relative.len()
        && import.module.as_ref().and_then(flatten) == wanted.module.as_ref().and_then(flatten)
}

fn imported_name_count(import: &ImportFrom<'_>, name: &str) -> usize {
    let ImportNames::Aliases(names) = &import.names else {
        return 0;
    };
    names
        .iter()
        .filter(|alias| {
            alias.asname.is_none()
                && flatten(&alias.name).is_some_and(|parts| parts.as_slice() == [name])
        })
        .count()
}

// Ensure an explicit top-level from-import exists. New imports are appended to
// the module body: package initializers commonly define names before their
// imports, so moving the import into a guessed "import block" can change
// initialization and circular-import behavior.
pub fn ensure_import_source(
    path: &str,
    src: &str,
    from: &str,
    name: &str,
) -> Result<Option<(String, usize)>> {
    validate_ensure_import(from, name)?;
    let wanted_text = format!("from {from} import {name}\n");
    let wanted_module = libcst_native::parse_module(&wanted_text, None).unwrap();
    let Statement::Simple(wanted_line) = &wanted_module.body[0] else {
        unreachable!();
    };
    let SmallStatement::ImportFrom(wanted) = &wanted_line.body[0] else {
        unreachable!();
    };

    let owned = parsed_module(path, src)?;
    let module = owned.borrow_dependent();
    let mut matches = 0;
    for statement in &module.body {
        let Statement::Simple(line) = statement else {
            continue;
        };
        for small in &line.body {
            if let SmallStatement::ImportFrom(import) = small
                && same_import_source(import, wanted)
            {
                matches += imported_name_count(import, name);
            }
        }
    }
    if matches > 1 {
        bail!(
            "{path}: `from {from} import {name}` is already satisfied by {matches} top-level imports; remove the duplicate"
        );
    }
    if matches == 1 {
        return Ok(None);
    }

    let mut state = CodegenState {
        default_newline: module.default_newline,
        default_indent: module.default_indent,
        ..Default::default()
    };
    for header in &module.header {
        header.codegen(&mut state);
    }
    for statement in &module.body {
        statement.codegen(&mut state);
    }
    let insert_at = state.tokens.len();
    if !src.starts_with(&state.tokens) {
        bail!("{path}: could not locate the end of the module body");
    }

    let newline = module.default_newline;
    let mut addition = String::new();
    if insert_at > 0 && !state.tokens.ends_with(['\n', '\r']) {
        addition.push_str(newline);
    }
    addition.push_str(&format!("from {from} import {name}"));
    if module.has_trailing_newline || !module.footer.is_empty() {
        addition.push_str(newline);
    }

    let mut result = src.to_string();
    result.insert_str(insert_at, &addition);
    if let Err(e) = cached_module(&result) {
        bail!("{path}: ensured import output no longer parses: {e}");
    }
    Ok(Some((result, 1)))
}

fn finish_rewrites(
    path: &str,
    src: &str,
    rewrites: Vec<Rewrite>,
    what: &str,
) -> Result<Option<(String, usize)>> {
    if rewrites.is_empty() {
        return Ok(None);
    }
    let count = rewrites.iter().map(|r| r.nodes).sum();
    let result = splice(path, src, rewrites)?;
    if let Err(e) = cached_module(&result) {
        bail!("{path}: {what} output no longer parses: {e}");
    }
    Ok(Some((result, count)))
}

// Boundary-aware, so `from pkg.utils import x` does not match inside
// `from pkg.utils_extra import x`.
fn statement_occurrences(text: &str, needle: &str) -> Vec<usize> {
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let bytes = text.as_bytes();
    let mut positions = Vec::new();
    let mut start = 0;
    while let Some(rel) = text[start..].find(needle) {
        let pos = start + rel;
        let before_ok = pos == 0 || !is_ident(bytes[pos - 1]);
        let end = pos + needle.len();
        let suffix = &text[end..];
        let after_ok = end == bytes.len()
            || (!is_ident(bytes[end])
                && bytes[end] != b'.'
                && bytes[end] != b','
                && !suffix.starts_with(" as "));
        if before_ok && after_ok {
            positions.push(pos);
        }
        start = pos + 1;
    }
    positions
}

// Rewrites are spliced into the text. Regenerating the module through libcst
// would reformat everything the rewrite did not touch.
fn splice(path: &str, src: &str, rewrites: Vec<Rewrite>) -> Result<String> {
    let mut out = src.to_string();
    for rw in rewrites {
        let positions = statement_occurrences(&out, &rw.old);
        if positions.len() != rw.nodes {
            bail!(
                "{path}: statement {:?} occurs {} time(s) in the text but \
                 {} time(s) as an import node; refusing to splice",
                rw.old,
                positions.len(),
                rw.nodes
            );
        }
        for pos in positions.into_iter().rev() {
            out.replace_range(pos..pos + rw.old.len(), &rw.new);
        }
    }
    Ok(out)
}

fn rewrite_from_imports(
    path: &str,
    src: &str,
    plan: impl Fn(&[&str]) -> Option<(usize, String)>,
) -> Result<Option<(String, usize)>> {
    let owned = parsed_module(path, src)?;
    let module = owned.borrow_dependent();
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for (import, indents) in module_imports(module).from_imports {
        if !import.relative.is_empty() {
            continue;
        }
        let Some(target) = import.module.as_ref().and_then(flatten) else {
            continue;
        };
        let Some((dots, new_module)) = plan(&target) else {
            continue;
        };
        add_rewrite(
            &mut rewrites,
            render_indented(import, &indents),
            render_with_module(import, dots, &new_module, &indents),
        );
    }
    finish_rewrites(path, src, rewrites, "rewritten")
}

pub fn relativize_source(
    path: &str,
    src: &str,
    package: &[impl AsRef<str>],
) -> Result<Option<(String, usize)>> {
    rewrite_from_imports(path, src, |target| {
        if target[0] != package[0].as_ref() {
            return None;
        }
        let k = package
            .iter()
            .zip(target.iter())
            .take_while(|(x, y)| x.as_ref() == **y)
            .count();
        // One dot for the file's own package, plus one per level climbed out
        // of it to reach the common ancestor.
        let dots = package.len() - k + 1;
        Some((dots, target[k..].join(".")))
    })
}

pub fn relativize_source_from_root(
    path: &str,
    src: &str,
    package: &[impl AsRef<str>],
) -> Result<Option<(String, usize)>> {
    rewrite_from_imports(path, src, |target| {
        if target[0] != package[0].as_ref() {
            return None;
        }
        Some((package.len(), target[1..].join(".")))
    })
}

pub fn remap_source(
    path: &str,
    src: &str,
    from_prefix: &DottedPath,
    to_prefix: &DottedPath,
) -> Result<Option<(String, usize)>> {
    let from = from_prefix.parts();
    rewrite_from_imports(path, src, |target| {
        if target.len() < from.len() || target[..from.len()] != *from {
            return None;
        }
        let mut new = to_prefix.parts().to_vec();
        new.extend(
            target[from.len()..]
                .iter()
                .map(std::string::ToString::to_string),
        );
        Some((0, new.join(".")))
    })
}

pub fn convert_imports_source(
    path: &str,
    src: &str,
    prefix: &DottedPath,
) -> Result<Option<(String, usize)>> {
    let prefix = prefix.parts();
    let owned = parsed_module(path, src)?;
    let module = owned.borrow_dependent();
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for (import, indents) in module_imports(module).plain_imports {
        let matching: Vec<_> = import
            .names
            .iter()
            .filter(|alias| {
                flatten(&alias.name)
                    .is_some_and(|t| t.len() >= prefix.len() && t[..prefix.len()] == *prefix)
            })
            .collect();
        if matching.is_empty() {
            continue;
        }
        if import.names.len() > 1 {
            bail!(
                "{path}: multi-name import statement mentions the prefix; split it \
                 with a replace before convert_import"
            );
        }
        let alias = matching[0];
        let target = flatten(&alias.name).unwrap();
        if target.len() < 2 {
            bail!(
                "{path}: `import {}` cannot be converted to a from-import; handle it \
                 with a replace",
                target.join(".")
            );
        }
        let Some(asname) = &alias.asname else {
            bail!(
                "{path}: `import {}` without an alias binds the top-level package; \
                 handle it with a replace",
                target.join(".")
            );
        };
        let libcst_native::AssignTargetExpression::Name(alias_name) = &asname.name else {
            bail!("{path}: unsupported import alias form");
        };
        let mut new = format!(
            "from {} import {} as {}",
            target[..target.len() - 1].join("."),
            target[target.len() - 1],
            alias_name.value
        );
        if let Some(semi) = &import.semicolon {
            new.push_str(&render(semi));
        }
        add_rewrite(&mut rewrites, render_indented(import, &indents), new);
    }
    finish_rewrites(path, src, rewrites, "converted")
}

fn import_alias_binding<'a>(path: &str, alias: &'a ImportAlias<'_>) -> Result<&'a str> {
    if let Some(asname) = &alias.asname {
        let AssignTargetExpression::Name(name) = &asname.name else {
            bail!("{path}: unsupported import alias form");
        };
        return Ok(name.value);
    }
    let Some(parts) = flatten(&alias.name) else {
        bail!("{path}: unsupported import name");
    };
    Ok(parts[0])
}

fn kernel_module_expr(helper: &str, suffix: &[&str]) -> String {
    if suffix.is_empty() {
        format!("{helper}()")
    } else {
        let suffix = serde_json::to_string(&suffix.join(".")).unwrap();
        format!("{helper}({suffix})")
    }
}

fn append_original_semicolon(text: &mut String, semicolon: Option<&libcst_native::Semicolon<'_>>) {
    if let Some(semicolon) = semicolon {
        text.push_str(&render(semicolon));
    }
}

fn kernelize_from_import(
    path: &str,
    import: &ImportFrom<'_>,
    package: &str,
    helper: &str,
) -> Result<Option<String>> {
    if !import.relative.is_empty() {
        return Ok(None);
    }
    let Some(target) = import.module.as_ref().and_then(flatten) else {
        return Ok(None);
    };
    if target.first() != Some(&package) {
        return Ok(None);
    }
    if import.lpar.is_some() || import.rpar.is_some() {
        bail!(
            "{path}: parenthesized import from `{}` is not supported by kernelize_imports; split it into a one-line import",
            target.join(".")
        );
    }
    let ImportNames::Aliases(aliases) = &import.names else {
        bail!(
            "{path}: wildcard import from `{}` cannot be kernelized safely",
            target.join(".")
        );
    };
    let module = kernel_module_expr(helper, &target[1..]);
    let mut bindings = Vec::with_capacity(aliases.len());
    let mut values = Vec::with_capacity(aliases.len());
    for alias in aliases {
        let Some(imported) = flatten(&alias.name) else {
            bail!("{path}: unsupported imported name");
        };
        if imported.len() != 1 {
            bail!("{path}: unsupported imported name {}", imported.join("."));
        }
        let binding = import_alias_binding(path, alias)?;
        let imported = serde_json::to_string(imported[0]).unwrap();
        bindings.push(binding);
        values.push(format!("getattr({module}, {imported})"));
    }
    let mut replacement = format!("{} = {}", bindings.join(", "), values.join(", "));
    append_original_semicolon(&mut replacement, import.semicolon.as_ref());
    Ok(Some(replacement))
}

fn kernelize_plain_import(
    path: &str,
    import: &Import<'_>,
    package: &str,
    helper: &str,
) -> Result<Option<String>> {
    let matching: Vec<_> = import
        .names
        .iter()
        .filter(|alias| flatten(&alias.name).is_some_and(|parts| parts.first() == Some(&package)))
        .collect();
    if matching.is_empty() {
        return Ok(None);
    }
    if import.names.len() != 1 {
        bail!(
            "{path}: a multi-name import statement mentions {package:?}; split it before kernelize_imports"
        );
    }
    let alias = matching[0];
    let target = flatten(&alias.name).unwrap();
    let binding = import_alias_binding(path, alias)?;
    let root = kernel_module_expr(helper, &[]);
    let mut replacement = if target.len() == 1 {
        format!("{binding} = {root}")
    } else {
        let module = kernel_module_expr(helper, &target[1..]);
        if alias.asname.is_some() {
            format!("{binding} = {module}")
        } else {
            // `import pkg.sub` binds pkg and also loads pkg.sub.
            format!("{binding} = {root}; {module}")
        }
    };
    append_original_semicolon(&mut replacement, import.semicolon.as_ref());
    Ok(Some(replacement))
}

fn is_module_docstring(statement: &Statement<'_>) -> bool {
    let Statement::Simple(line) = statement else {
        return false;
    };
    let [SmallStatement::Expr(expr)] = line.body.as_slice() else {
        return false;
    };
    matches!(
        expr.value,
        Expression::SimpleString(_) | Expression::ConcatenatedString(_)
    )
}

fn is_future_import(statement: &Statement<'_>) -> bool {
    let Statement::Simple(line) = statement else {
        return false;
    };
    let [SmallStatement::ImportFrom(import)] = line.body.as_slice() else {
        return false;
    };
    import.relative.is_empty()
        && import.module.as_ref().and_then(flatten).as_deref() == Some(["__future__"].as_slice())
}

fn insert_kernel_helper(
    path: &str,
    src: &str,
    helper: &str,
    kernel: &str,
    version: usize,
) -> Result<String> {
    let owned = parsed_module(path, src)?;
    let module = owned.borrow_dependent();
    let mut body_index = usize::from(module.body.first().is_some_and(is_module_docstring));
    while module.body.get(body_index).is_some_and(is_future_import) {
        body_index += 1;
    }

    let mut state = CodegenState {
        default_newline: module.default_newline,
        default_indent: module.default_indent,
        ..Default::default()
    };
    for header in &module.header {
        header.codegen(&mut state);
    }
    for statement in &module.body[..body_index] {
        statement.codegen(&mut state);
    }
    let insert_at = state.tokens.len();
    if !src.starts_with(&state.tokens) {
        bail!("{path}: could not locate the kernel import helper insertion point");
    }

    let newline = module.default_newline;
    let indent = module.default_indent;
    let kernel = serde_json::to_string(kernel).unwrap();
    let mut helper_source = String::new();
    if insert_at > 0 && !state.tokens.ends_with(['\n', '\r']) {
        helper_source.push_str(newline);
    }
    let cached_root = format!("{helper}_root");
    helper_source.push_str(&format!("{cached_root} = None{newline}"));
    helper_source.push_str(&format!("def {helper}(module=\"\"):{newline}"));
    helper_source.push_str(&format!("{indent}global {cached_root}{newline}"));
    helper_source.push_str(&format!("{indent}if {cached_root} is None:{newline}"));
    helper_source.push_str(&format!(
        "{indent}{indent}{cached_root} = __import__(\"kernels\").get_kernel({kernel}, version={version}){newline}"
    ));
    helper_source.push_str(&format!("{indent}root = {cached_root}{newline}"));
    helper_source.push_str(&format!("{indent}if not module:{newline}"));
    helper_source.push_str(&format!("{indent}{indent}return root{newline}"));
    helper_source.push_str(&format!(
        "{indent}return __import__(\"importlib\").import_module(root.__name__ + \".\" + module){newline}{newline}"
    ));

    let mut result = src.to_string();
    result.insert_str(insert_at, &helper_source);
    check_parses(path, &result)?;
    Ok(result)
}

pub fn kernelize_imports_source(
    path: &str,
    src: &str,
    package: &str,
    kernel: &str,
    version: usize,
) -> Result<Option<(String, usize)>> {
    let parsed_package: DottedPath = package.parse()?;
    if parsed_package.parts().len() != 1 {
        bail!("kernelized package must be one top-level Python name, got {package:?}");
    }
    if kernel.is_empty() {
        bail!("kernel must not be empty");
    }

    let helper_base = format!("__kernel_port_{package}");
    let mut helper = helper_base.clone();
    let mut suffix = 2;
    while src.contains(&helper) {
        helper = format!("{helper_base}_{suffix}");
        suffix += 1;
    }

    let owned = parsed_module(path, src)?;
    let module = owned.borrow_dependent();
    let imports = module_imports(module);
    let mut rewrites = Vec::new();
    for (import, indents) in imports.from_imports {
        if let Some(new) = kernelize_from_import(path, import, package, &helper)? {
            add_rewrite(&mut rewrites, render_indented(import, &indents), new);
        }
    }
    for (import, indents) in imports.plain_imports {
        if let Some(new) = kernelize_plain_import(path, import, package, &helper)? {
            add_rewrite(&mut rewrites, render_indented(import, &indents), new);
        }
    }
    let Some((result, count)) = finish_rewrites(path, src, rewrites, "kernelized")? else {
        return Ok(None);
    };
    let result = insert_kernel_helper(path, &result, &helper, kernel, version)?;
    let remaining = absolute_self_imports(path, &result, package)?;
    if !remaining.is_empty() {
        bail!(
            "{path}: kernelize_imports left matching static import(s): {}",
            remaining.join(", ")
        );
    }
    Ok(Some((result, count)))
}

pub fn absolute_self_imports(path: &str, src: &str, package: &str) -> Result<Vec<String>> {
    let key = (memo::digest(src), package.to_string());
    if let Some(found) = memo::self_imports(&key) {
        return Ok(found);
    }
    let owned = module_of(path, src)?;
    let imports = module_imports(owned.borrow_dependent());

    let mut offenders = Vec::new();
    for (import, _) in imports.from_imports {
        if !import.relative.is_empty() {
            continue;
        }
        if let Some(target) = import.module.as_ref().and_then(flatten)
            && target[0] == package
        {
            offenders.push(format!("from {} import ...", target.join(".")));
        }
    }
    for (import, _) in imports.plain_imports {
        for alias in &import.names {
            if let Some(target) = flatten(&alias.name)
                && target[0] == package
            {
                offenders.push(format!("import {}", target.join(".")));
            }
        }
    }
    memo::note_self_imports(key, offenders.clone());
    Ok(offenders)
}

pub fn check_parses(path: &str, src: &str) -> Result<()> {
    if memo::parses(memo::digest(src)) {
        return Ok(());
    }
    if let Err(e) = cached_module(src) {
        bail!("verify: {path} does not parse as Python: {e}");
    }
    Ok(())
}
