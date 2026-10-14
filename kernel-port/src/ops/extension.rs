// Turn an upstream native extension into a kernel-builder one. The recipe
// states what the upstream ships (its C++ sources, its Torch op namespace, the
// Python modules it is imported as) and this op derives the edits that every
// such port needs, instead of each recipe spelling them out.

use crate::python;
use crate::recipe::Args;
use crate::rewrite::{Lang, Rule, attribute_uses};
use crate::workspace::{Pattern, Workspace};
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;

pub struct Extension {
    sources: Option<Pattern>,
    python: Pattern,
    namespace: Option<String>,
    modules: Vec<String>,
    changes: usize,
}

impl std::fmt::Debug for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extension")
            .field("sources", &self.sources)
            .field("namespace", &self.namespace)
            .field("modules", &self.modules)
            .finish_non_exhaustive()
    }
}

// Where kernel-builder generates the registration helpers.
const REGISTRATION: &str = "torch-ext/registration.h";

const TORCH_LIBRARY_MACROS: &[&str] = &[
    "STABLE_TORCH_LIBRARY_IMPL",
    "STABLE_TORCH_LIBRARY_FRAGMENT",
    "STABLE_TORCH_LIBRARY",
    "TORCH_LIBRARY_IMPL",
    "TORCH_LIBRARY_FRAGMENT",
    "TORCH_LIBRARY",
];

fn is_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !s.as_bytes()[0].is_ascii_digit()
}

#[derive(Default)]
struct Tally(BTreeMap<&'static str, usize>);

impl Tally {
    fn add(&mut self, what: &'static str, n: usize) {
        if n > 0 {
            *self.0.entry(what).or_default() += n;
        }
    }
    fn total(&self) -> usize {
        self.0.values().sum()
    }
    fn describe(&self) -> String {
        if self.0.is_empty() {
            return "nothing".into();
        }
        self.0
            .iter()
            .map(|(k, v)| format!("{v} {k}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Extension {
    pub(super) fn build(args: &mut Args) -> Result<Self> {
        let sources = args.take_opt("sources").map(|s| s.parse()).transpose()?;
        let python = args
            .take_opt("python")
            .unwrap_or_else(|| "torch-ext/**/*.py".into())
            .parse()?;
        let namespace = args.take_opt("namespace");
        if let Some(ns) = &namespace
            && !is_identifier(ns)
        {
            bail!("namespace {ns:?} must be a C++ identifier");
        }
        let modules: Vec<String> = args
            .take_opt("modules")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        for m in &modules {
            if !m.split('.').all(is_identifier) {
                bail!("module {m:?} must be a dotted Python module name");
            }
        }
        if sources.is_none() && namespace.is_none() && modules.is_empty() {
            bail!("extension needs at least one of sources=, namespace= or modules=");
        }
        let changes = args.take_usize("changes")?;
        Ok(Self {
            sources,
            python,
            namespace,
            modules,
            changes,
        })
    }

    pub(super) fn apply(&self, ws: &mut Workspace) -> Result<String> {
        let mut tally = Tally::default();
        let mut updates: Vec<(String, String)> = Vec::new();
        if let Some(sources) = &self.sources {
            let paths = ws.glob(sources);
            if paths.is_empty() {
                bail!("sources {sources:?} matches no files");
            }
            for path in paths {
                let text = ws.get_text(&path)?;
                let new = self.convert_cpp(&path, text, &mut tally)?;
                if new != text {
                    updates.push((path, new));
                }
            }
        }
        if self.namespace.is_some() || !self.modules.is_empty() {
            for path in ws.glob(&self.python) {
                if !crate::is_python(&path) {
                    continue;
                }
                let text = ws.get_text(&path)?;
                let new = self.convert_python(&path, text, &mut tally)?;
                if new != text {
                    updates.push((path, new));
                }
            }
        }
        if tally.total() != self.changes {
            bail!(
                "expected exactly {} change(s) but made {} ({}) - upstream drifted; \
                 review them and update changes=",
                self.changes,
                tally.total(),
                tally.describe()
            );
        }
        for (path, text) in updates {
            ws.set_text(&path, text);
        }
        Ok(format!("converted the extension: {}", tally.describe()))
    }

    fn convert_cpp(&self, path: &str, text: &str, tally: &mut Tally) -> Result<String> {
        let lang = Lang::CFamily;
        let mut text = text.to_string();
        let mut run =
            |pattern: &str, with: &str, what: &'static str, text: &mut String| -> Result<usize> {
                let (new, n) = Rule::new(pattern, with)?.apply(text, lang);
                *text = new;
                tally.add(what, n);
                Ok(n)
            };

        // Ops are registered through TORCH_LIBRARY, not a pybind11 module.
        let removed = run(
            "PYBIND11_MODULE(:[name], :[m]) {:[defs]}",
            "",
            "pybind module(s) removed",
            &mut text,
        )?;
        // A module at the end of the file leaves the blank line before it.
        if removed > 0 && text.ends_with("\n\n") {
            text.truncate(text.trim_end_matches('\n').len() + 1);
        }
        for header in [
            "<pybind11/pybind11.h>",
            "<pybind11/stl.h>",
            "<pybind11/numpy.h>",
        ] {
            run(
                &format!("#include {header}"),
                "",
                "pybind include(s) removed",
                &mut text,
            )?;
        }
        // torch/extension.h and torch/python.h pull in the Python bindings;
        // the first becomes torch/all.h unless the file already has it.
        let mut have_all = text.contains("<torch/all.h>") || text.contains("\"torch/all.h\"");
        for header in [
            "<torch/extension.h>",
            "\"torch/extension.h\"",
            "<torch/python.h>",
            "\"torch/python.h\"",
        ] {
            let with = if have_all {
                ""
            } else {
                "#include <torch/all.h>"
            };
            let n = run(
                &format!("#include {header}"),
                with,
                "torch include(s) swapped",
                &mut text,
            )?;
            have_all |= n > 0;
        }

        if let Some(ns) = &self.namespace {
            let mut converted = 0;
            for macro_name in TORCH_LIBRARY_MACROS {
                let with = format!("{macro_name}_EXPAND(TORCH_EXTENSION_NAME, :[rest])");
                let (new, n) =
                    Rule::new(&format!("{macro_name}({ns}, :[rest])"), &with)?.apply(&text, lang);
                text = new;
                converted += n;
            }
            if converted > 0 {
                tally.add("TORCH_LIBRARY registration(s) namespaced", converted);
                let include = format!("#include \"{}\"", crate::cpp::relative(path, REGISTRATION));
                if !text.contains(&include) {
                    let at = TORCH_LIBRARY_MACROS
                        .iter()
                        .filter_map(|m| text.find(&format!("{m}_EXPAND(")))
                        .min()
                        .expect("a registration was just converted");
                    let line = text[..at].rfind('\n').map_or(0, |i| i + 1);
                    text.insert_str(line, &format!("{include}\n\n"));
                }
            }
        }

        for leftover in ["PYBIND11_", "py::"] {
            let (_, n) = Rule::new(leftover, leftover)?.apply(&text, lang);
            if n > 0 {
                bail!(
                    "{path} still uses pybind11 ({leftover}) after conversion; the binding \
                     is not a plain module of m.def calls - overlay the file or rewrite it"
                );
            }
        }
        Ok(text)
    }

    fn convert_python(&self, path: &str, text: &str, tally: &mut Tally) -> Result<String> {
        let lang = Lang::Python;
        let package_dots = package_dots(path)
            .with_context(|| format!("{path} is not inside a torch-ext/<package> directory"))?;
        let mut text = text.to_string();
        let mut needs: Vec<&str> = Vec::new();

        for module in &self.modules {
            let top = module.split('.').next().unwrap();
            let import = format!("import {module}");
            let (_, imports) = Rule::new(&import, &import)?.apply(&text, lang);
            if imports == 0 {
                let (_, from_imports) =
                    Rule::new(&format!("from {module} import"), "x")?.apply(&text, lang);
                if from_imports > 0 {
                    bail!(
                        "{path}: `from {module} import ...` is not supported; rewrite it to `import {module}`"
                    );
                }
                // Upstream may build the module at import time instead.
                let import_line = format!("from {package_dots}_ops import ops as {{var}}");
                if let Some((new, loads, removed)) =
                    python::replace_jit_loads(path, &text, module, &import_line)?
                {
                    text = new;
                    tally.add("JIT build(s) replaced", loads);
                    tally.add("JIT build setting(s) removed", removed);
                }
                continue;
            }
            // An attribute access through the module, outside its own imports,
            // means it is used as an object rather than only imported for its
            // side effect of registering ops.
            let (without_imports, _) = Rule::new(&import, "")?.apply(&text, lang);
            let used = attribute_uses(&without_imports, lang, top) > 0;
            let with = if used {
                if module.contains('.') {
                    bail!(
                        "{path}: dotted extension module {module} is used as an object; rewrite its uses first"
                    );
                }
                format!("from {package_dots}_ops import ops as {module}")
            } else {
                format!("from {package_dots} import _ops")
            };
            let (new, n) = Rule::new(&import, &with)?.apply(&text, lang);
            text = new;
            tally.add("extension import(s) redirected", n);
        }

        if let Some(ns) = &self.namespace {
            let (new, n) = Rule::new(&format!("torch.ops.{ns}"), "ops")?.apply(&text, lang);
            text = new;
            if n > 0 {
                tally.add("torch.ops namespace use(s) redirected", n);
                needs.push("ops");
            }
            let (new, n) = Rule::new(
                &format!("\"{ns}:::[[name]]\""),
                "add_op_namespace_prefix(\":[name]\")",
            )?
            .apply(&text, lang);
            text = new;
            if n > 0 {
                tally.add("op name(s) prefixed", n);
                needs.push("add_op_namespace_prefix");
            }
            // An op schema literal ("ns::name(Tensor x) -> ...") also names an
            // op, but splicing the prefix into it depends on how the string is
            // built; leave that to an explicit rewrite rather than guess.
            let (_, left) =
                Rule::new(&format!("\"{ns}:::[[name]](:[rest]\""), "x")?.apply(&text, lang);
            if left > 0 {
                bail!(
                    "{path}: {left} op schema string(s) still name the {ns} namespace; \
                     rewrite them explicitly"
                );
            }
        }

        if !needs.is_empty() {
            needs.sort_unstable();
            needs.dedup();
            text = insert_import(
                &text,
                &format!("from {package_dots}_ops import {}", needs.join(", ")),
            );
        }
        Ok(text)
    }
}

// The relative-import prefix from a module under torch-ext/<package>/ back to
// the package root: "." at the root, ".." one directory down, and so on.
fn package_dots(path: &str) -> Option<String> {
    let rest = path.strip_prefix("torch-ext/")?;
    let depth = rest.matches('/').count();
    (depth >= 1).then(|| ".".repeat(depth))
}

// Insert `line` after the module's leading block of imports, keeping a
// docstring, comments and `__future__` imports above it.
fn insert_import(text: &str, line: &str) -> String {
    let mut offset = 0;
    let mut insert_at = None;
    let mut depth = 0i32;
    let mut in_doc = false;
    let mut seen_code = false;
    for l in text.split_inclusive('\n') {
        let t = l.trim();
        offset += l.len();
        if in_doc {
            if t.contains("\"\"\"") || t.contains("'''") {
                in_doc = false;
            }
            continue;
        }
        if depth > 0 {
            depth += paren_delta(t);
            if depth <= 0 {
                insert_at = Some(offset);
            }
            continue;
        }
        if !seen_code && (t.starts_with("\"\"\"") || t.starts_with("'''")) {
            let q = &t[..3];
            in_doc = t.len() < 6 || !t[3..].contains(q);
            seen_code = true;
            continue;
        }
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let is_import = (l.starts_with("import ") || l.starts_with("from ")) && !l.starts_with(' ');
        if is_import {
            seen_code = true;
            depth = paren_delta(t);
            if depth <= 0 {
                insert_at = Some(offset);
            }
            continue;
        }
        break;
    }
    let at = insert_at.unwrap_or(0);
    let mut out = String::with_capacity(text.len() + line.len() + 2);
    out.push_str(&text[..at]);
    if at > 0 && !text[..at].ends_with('\n') {
        out.push('\n');
    }
    out.push_str(line);
    out.push('\n');
    out.push_str(&text[at..]);
    out
}

fn paren_delta(line: &str) -> i32 {
    let code = line.split('#').next().unwrap_or("");
    let open = code.matches('(').count();
    let close = code.matches(')').count();
    i32::try_from(open).unwrap_or(0) - i32::try_from(close).unwrap_or(0)
}
