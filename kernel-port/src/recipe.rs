use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct Invocation {
    pub line: usize,
    pub op: String,
    pub args: BTreeMap<String, String>,
}

impl Invocation {
    pub fn take_args(&self) -> Args<'_> {
        Args {
            op: &self.op,
            map: self.args.clone(),
        }
    }
}

pub struct Args<'a> {
    op: &'a str,
    map: BTreeMap<String, String>,
}

impl Args<'_> {
    pub fn take(&mut self, key: &str) -> Result<String> {
        self.map
            .remove(key)
            .with_context(|| format!("op {:?} requires argument {key}=\"...\"", self.op))
    }

    pub fn take_opt(&mut self, key: &str) -> Option<String> {
        self.map.remove(key)
    }

    pub fn take_usize(&mut self, key: &str) -> Result<usize> {
        self.take_usize_opt(key)?
            .with_context(|| format!("op {:?} requires argument {key}=\"...\"", self.op))
    }

    pub fn take_usize_opt(&mut self, key: &str) -> Result<Option<usize>> {
        match self.map.remove(key) {
            Some(v) => Ok(Some(v.parse().with_context(|| {
                format!("argument {key} of op {:?} must be an integer", self.op)
            })?)),
            None => Ok(None),
        }
    }

    pub fn finish(self) -> Result<()> {
        if let Some(key) = self.map.keys().next() {
            bail!("op {:?} does not take an argument named {key:?}", self.op);
        }
        Ok(())
    }
}

// Bumped when a change gives an existing recipe a different meaning, not when
// it adds a capability.
pub const VERSION: u64 = 1;

pub struct Recipe {
    pub version: Option<u64>,
    pub ops: Vec<Invocation>,
}

impl Recipe {
    pub const fn effective_version(&self) -> u64 {
        match self.version {
            Some(v) => v,
            None => VERSION,
        }
    }
}

pub fn parse(text: &str) -> Result<Recipe> {
    let doc = kdl::KdlDocument::parse_v2(text).map_err(|e| format_parse_error(&e))?;
    let mut out = Vec::new();
    for node in doc.nodes() {
        let line = line_of(text, node.span().offset());
        let op = node.name().value().to_string();
        if node.ty().is_some() {
            bail!("recipe line {line}: {op}: type annotations are not part of the recipe language");
        }
        if node.children().is_some() {
            bail!("recipe line {line}: {op}: children blocks are not part of the recipe language");
        }
        let mut args = BTreeMap::new();
        for entry in node.entries() {
            let Some(name) = entry.name() else {
                bail!(
                    "recipe line {line}: {op}: positional arguments are not allowed; \
                     every argument is key=value"
                );
            };
            let key = name.value().to_string();
            if entry.ty().is_some() {
                bail!(
                    "recipe line {line}: {op}: type annotations are not part of the \
                     recipe language"
                );
            }
            let value = match entry.value() {
                kdl::KdlValue::String(s) => s.clone(),
                kdl::KdlValue::Integer(n) => n.to_string(),
                kdl::KdlValue::Bool(b) => b.to_string(),
                other => bail!(
                    "recipe line {line}: {op}: argument {key} must be a string, \
                     integer, or #true/#false, got {other}"
                ),
            };
            // KDL takes the rightmost of a repeated property; here that would
            // swallow a typo.
            if args.insert(key.clone(), value).is_some() {
                bail!("recipe line {line}: duplicate argument {key:?}");
            }
        }
        out.push(Invocation { line, op, args });
    }
    let version = take_version_header(&mut out)?;
    Ok(Recipe { version, ops: out })
}

// The header is not an op: it declares the format the rest of the file is
// written in, so it means nothing anywhere but the first node.
fn take_version_header(ops: &mut Vec<Invocation>) -> Result<Option<u64>> {
    if let Some(stray) = ops.iter().skip(1).find(|inv| inv.op == "recipe") {
        bail!(
            "recipe line {}: the `recipe version=N` header must be the first \
             node in the file",
            stray.line
        );
    }
    if ops.first().is_none_or(|inv| inv.op != "recipe") {
        return Ok(None);
    }
    let header = ops.remove(0);
    let mut args = header.take_args();
    let raw = args.take("version")?;
    args.finish()?;
    let version: u64 = raw
        .parse()
        .with_context(|| format!("recipe version must be a positive integer, got {raw:?}"))?;
    if version == 0 || version > VERSION {
        bail!(
            "recipe declares version {version}, but this kernel-port implements \
             version {VERSION} - upgrade kernel-port, or re-verify the recipe \
             against this version and update its header"
        );
    }
    Ok(Some(version))
}

// KDL reports char offsets. Recipes are ASCII, where chars and bytes coincide.
fn line_of(src: &str, offset: usize) -> usize {
    src.chars().take(offset).filter(|&c| c == '\n').count() + 1
}

fn format_parse_error(err: &kdl::KdlError) -> anyhow::Error {
    let mut lines = vec!["recipe parse error:".to_string()];
    for d in &err.diagnostics {
        let line = line_of(&err.input, d.span.offset());
        let msg = d.message.as_deref().unwrap_or("invalid syntax");
        match &d.help {
            Some(help) => lines.push(format!("  line {line}: {msg} ({help})")),
            None => lines.push(format!("  line {line}: {msg}")),
        }
    }
    anyhow::anyhow!(lines.join("\n"))
}
