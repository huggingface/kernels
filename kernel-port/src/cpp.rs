// C, C++ and CUDA source handling: which files are sources, and their quoted
// `#include` directives.

use std::collections::BTreeMap;

const EXTENSIONS: &[&str] = &[
    "c", "cc", "cpp", "cxx", "cu", "cuh", "h", "hh", "hpp", "hxx", "inl", "inc", "mm", "metal",
];

pub(crate) fn is_source(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTENSIONS.contains(&e))
}

/// A quoted include: the byte range of the path between the quotes.
pub(crate) struct Include {
    pub start: usize,
    pub end: usize,
}

/// Quoted `#include "..."` directives, one per line. Angle-bracket includes
/// name system or include-path headers and are not file references.
pub(crate) fn quoted_includes(text: &str) -> Vec<Include> {
    let mut out = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let rest = line.trim_start_matches([' ', '\t']);
        let after_hash = rest
            .strip_prefix('#')
            .map(|r| r.trim_start_matches([' ', '\t']));
        if let Some(r) = after_hash.and_then(|r| r.strip_prefix("include")) {
            let r2 = r.trim_start_matches([' ', '\t']);
            if let Some(body) = r2.strip_prefix('"')
                && let Some(close) = body.find('"')
            {
                let start = offset + (line.len() - body.len());
                out.push(Include {
                    start,
                    end: start + close,
                });
            }
        }
        offset += line.len();
    }
    out
}

fn dir_of(path: &str) -> &str {
    path.rfind('/').map_or("", |i| &path[..i])
}

/// Resolve `include` against the including file's directory, the first place
/// a compiler looks for a quoted include. None when it climbs above the root.
pub(crate) fn resolve(includer: &str, include: &str) -> Option<String> {
    let mut parts: Vec<&str> = dir_of(includer)
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    for seg in include.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

/// The shortest relative path from `from_file`'s directory to `to`.
pub(crate) fn relative(from_file: &str, to: &str) -> String {
    let from: Vec<&str> = dir_of(from_file)
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let to_parts: Vec<&str> = to.split('/').collect();
    let common = from
        .iter()
        .zip(&to_parts)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out: Vec<&str> = vec![".."; from.len() - common];
    out.extend(&to_parts[common..]);
    out.join("/")
}

/// Rewrite quoted includes broken by `moves` (old path -> new path). Each
/// include is resolved against the includer's pre-move location; when that
/// resolves to a file that existed and the includer or the target moved, it is
/// rewritten to the target's new location relative to the includer's new one.
/// Returns new texts keyed by current path, and the number of rewritten includes.
pub(crate) fn fix_includes(
    files: &BTreeMap<String, Vec<u8>>,
    old_paths: &std::collections::BTreeSet<String>,
    moves: &BTreeMap<String, String>,
) -> (Vec<(String, String)>, usize) {
    let old_of: BTreeMap<&str, &str> = moves
        .iter()
        .map(|(o, n)| (n.as_str(), o.as_str()))
        .collect();
    let mut updates = Vec::new();
    let mut count = 0;
    for (path, bytes) in files {
        if !is_source(path) {
            continue;
        }
        let Ok(text) = std::str::from_utf8(bytes) else {
            continue;
        };
        let old_includer = old_of.get(path.as_str()).copied().unwrap_or(path);
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        let mut n = 0;
        for inc in quoted_includes(text) {
            let target = &text[inc.start..inc.end];
            let Some(old_target) = resolve(old_includer, target) else {
                continue;
            };
            if !old_paths.contains(&old_target) {
                continue;
            }
            let new_target = moves.get(&old_target).unwrap_or(&old_target);
            if old_includer == path.as_str() && new_target == &old_target {
                continue;
            }
            let rewritten = relative(path, new_target);
            if rewritten == target {
                continue;
            }
            out.push_str(&text[last..inc.start]);
            out.push_str(&rewritten);
            last = inc.end;
            n += 1;
        }
        if n > 0 {
            out.push_str(&text[last..]);
            updates.push((path.clone(), out));
            count += n;
        }
    }
    (updates, count)
}

/// Every `#include` target in `text`, with whether it is quoted.
fn includes(text: &str) -> Vec<(&str, bool)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let rest = line.trim_start_matches([' ', '\t']);
        let Some(r) = rest.strip_prefix('#') else {
            continue;
        };
        let Some(r) = r.trim_start_matches([' ', '\t']).strip_prefix("include") else {
            continue;
        };
        let r = r.trim_start_matches([' ', '\t']);
        let (close, quoted) = match r.as_bytes().first() {
            Some(b'"') => ('"', true),
            Some(b'<') => ('>', false),
            _ => continue,
        };
        if let Some(end) = r[1..].find(close) {
            out.push((&r[1..=end], quoted));
        }
    }
    out
}

// `dir`/`include`, normalized; None when it climbs above the root.
fn join(dir: &str, include: &str) -> Option<String> {
    let mut parts: Vec<&str> = dir
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    for seg in include.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            s => parts.push(s),
        }
    }
    Some(parts.join("/"))
}

/// The files reachable from `roots` through `#include`s, roots included. A
/// quoted include is looked up next to the including file and then in
/// `include_dirs`, an angle-bracket include only in `include_dirs`, the way
/// a compiler searches them. Includes that name no file in `files` (system
/// and dependency headers) are skipped. Conditional compilation is not
/// evaluated, so this can only over-approximate.
pub(crate) fn include_closure<'a>(
    files: &BTreeMap<String, Vec<u8>>,
    roots: impl IntoIterator<Item = &'a String>,
    include_dirs: &[String],
) -> std::collections::BTreeSet<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack: Vec<String> = roots.into_iter().cloned().collect();
    while let Some(path) = stack.pop() {
        if !seen.insert(path.clone()) || !is_source(&path) {
            continue;
        }
        let Some(text) = files.get(&path).and_then(|b| std::str::from_utf8(b).ok()) else {
            continue;
        };
        for (include, quoted) in includes(text) {
            let local = quoted.then(|| join(dir_of(&path), include)).flatten();
            let found = local
                .into_iter()
                .chain(include_dirs.iter().filter_map(|d| join(d, include)))
                .find(|p| files.contains_key(p));
            if let Some(found) = found {
                stack.push(found);
            }
        }
    }
    seen
}
