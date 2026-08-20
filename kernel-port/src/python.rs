// Python edits go through libcst, so comments, quoting and layout outside the
// rewritten import survive byte-for-byte.
use anyhow::{Context, Result, bail};
use libcst_native::{
    Codegen, CodegenState, CompoundStatement, Expression, Import, ImportFrom, Module,
    NameOrAttribute, OrElse, SmallStatement, Statement, Suite,
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

fn parsed_module<'a>(path: &str, src: &'a str) -> Result<Module<'a>> {
    let module = libcst_native::parse_module(src, None)
        .map_err(|e| anyhow::anyhow!("parsing {path}: {e}"))?;
    check_roundtrip(&module, src).with_context(|| path.to_string())?;
    Ok(module)
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
    if let Err(e) = libcst_native::parse_module(&result, None) {
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
        let after_ok = end == bytes.len() || !is_ident(bytes[end]);
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
    let module = parsed_module(path, src)?;
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for (import, indents) in module_imports(&module).from_imports {
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
    let module = parsed_module(path, src)?;
    let mut rewrites: Vec<Rewrite> = Vec::new();
    for (import, indents) in module_imports(&module).plain_imports {
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

pub fn absolute_self_imports(path: &str, src: &str, package: &str) -> Result<Vec<String>> {
    let module = libcst_native::parse_module(src, None)
        .map_err(|e| anyhow::anyhow!("parsing {path}: {e}"))?;
    let imports = module_imports(&module);

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
    Ok(offenders)
}

pub fn check_parses(path: &str, src: &str) -> Result<()> {
    if let Err(e) = libcst_native::parse_module(src, None) {
        bail!("verify: {path} does not parse as Python: {e}");
    }
    Ok(())
}
