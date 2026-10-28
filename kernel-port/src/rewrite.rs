// Structural rewriting in the style of Comby: a pattern is literal text with
// holes, matched against source that has been lexed just enough to know where
// strings and comments are and how brackets nest. That makes matches immune to
// the failure modes of plain text replacement: they cannot land inside a
// string or comment, cannot cut an identifier in half, and holes always cover
// balanced text.

use anyhow::{Result, bail};
use std::collections::BTreeMap;

// Hole name -> the source range it matched.
type Binds = BTreeMap<String, (usize, usize)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lang {
    Python,
    CFamily,
    Plain,
}

impl Lang {
    pub(crate) fn of(path: &str) -> Self {
        if crate::is_python(path) {
            Self::Python
        } else if crate::cpp::is_source(path) {
            Self::CFamily
        } else {
            Self::Plain
        }
    }
}

// A string literal or a comment. For strings, `body_end` is where the closing
// delimiter starts; a hole that begins inside the literal stops there.
#[derive(Clone, Copy, Debug)]
struct Atom {
    start: usize,
    end: usize,
    body_end: usize,
}

struct Lexed {
    atoms: Vec<Atom>,
    // For each byte: the atom containing it.
    atom_at: Vec<Option<usize>>,
    // For each opening bracket in code: the index just past its closer.
    close_of: BTreeMap<usize, usize>,
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn lex(src: &[u8], lang: Lang) -> Lexed {
    let atoms = lex_atoms(src, lang);
    let n = src.len();
    let mut atom_at = vec![None; n];
    for (k, a) in atoms.iter().enumerate() {
        for slot in &mut atom_at[a.start..a.end] {
            *slot = Some(k);
        }
    }
    let close_of = match_brackets(src, &atom_at);
    Lexed {
        atoms,
        atom_at,
        close_of,
    }
}

// Strings and comments, in source order.
fn lex_atoms(src: &[u8], lang: Lang) -> Vec<Atom> {
    let mut atoms = Vec::new();
    let mut i = 0;
    let n = src.len();
    let starts_with = |i: usize, s: &[u8]| src[i..].starts_with(s);
    while i < n && lang != Lang::Plain {
        let c = src[i];
        let line_comment = match lang {
            Lang::Python => c == b'#',
            _ => starts_with(i, b"//"),
        };
        if line_comment {
            let end = src[i..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(n, |p| i + p);
            atoms.push(Atom {
                start: i,
                end,
                body_end: end,
            });
            i = end;
            continue;
        }
        if lang == Lang::CFamily && starts_with(i, b"/*") {
            let end = src[i + 2..]
                .windows(2)
                .position(|w| w == b"*/")
                .map_or(n, |p| i + 2 + p + 2);
            atoms.push(Atom {
                start: i,
                end,
                body_end: end.saturating_sub(2).max(i + 2),
            });
            i = end;
            continue;
        }
        if c == b'"' || c == b'\'' {
            // In C, a quote after an identifier character is a digit separator
            // (1'000) or a literal prefix handled as code.
            if lang == Lang::CFamily && c == b'\'' && i > 0 && src[i - 1].is_ascii_alphanumeric() {
                i += 1;
                continue;
            }
            let triple =
                lang == Lang::Python && (starts_with(i, b"\"\"\"") || starts_with(i, b"'''"));
            let delim: &[u8] = if triple { &src[i..i + 3] } else { &src[i..=i] };
            let mut j = i + delim.len();
            let mut body_end = n;
            let mut end = n;
            while j < n {
                if src[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if !triple && src[j] == b'\n' {
                    // Unterminated single-line literal: stop at the newline.
                    body_end = j;
                    end = j;
                    break;
                }
                if src[j..].starts_with(delim) {
                    body_end = j;
                    end = j + delim.len();
                    break;
                }
                j += 1;
            }
            atoms.push(Atom {
                start: i,
                end: end.min(n),
                body_end: body_end.min(n),
            });
            i = end.min(n).max(i + 1);
            continue;
        }
        i += 1;
    }

    atoms
}

// For each opening bracket in code, the index just past its closer.
fn match_brackets(src: &[u8], atom_at: &[Option<usize>]) -> BTreeMap<usize, usize> {
    let mut close_of = BTreeMap::new();
    let mut stack: Vec<(u8, usize)> = Vec::new();
    for (i, &c) in src.iter().enumerate() {
        if atom_at[i].is_some() {
            continue;
        }
        match c {
            b'(' | b'[' | b'{' => stack.push((c, i)),
            b')' | b']' | b'}' => {
                let open = match c {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                if let Some(pos) = stack.iter().rposition(|&(o, _)| o == open) {
                    let (_, at) = stack[pos];
                    stack.truncate(pos);
                    close_of.insert(at, i + 1);
                }
            }
            _ => {}
        }
    }
    close_of
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Lit(Vec<u8>),
    Ws,
    Hole(String),
    Ident(String),
}

#[derive(Debug)]
pub(crate) struct Template {
    pieces: Vec<Piece>,
}

fn parse_hole(s: &str, i: usize) -> Option<(Piece, usize)> {
    let rest = &s[i..];
    let (ident, open, close) = if rest.starts_with(":[[") {
        (true, 3, "]]")
    } else if rest.starts_with(":[") {
        (false, 2, "]")
    } else {
        return None;
    };
    let end = rest[open..].find(close)?;
    let name = &rest[open..open + end];
    if name.is_empty() || !name.bytes().all(is_ident) {
        return None;
    }
    let piece = if ident {
        Piece::Ident(name.into())
    } else {
        Piece::Hole(name.into())
    };
    Some((piece, open + end + close.len()))
}

impl Template {
    pub(crate) fn parse(s: &str, collapse_ws: bool) -> Self {
        let mut pieces: Vec<Piece> = Vec::new();
        let mut lit = Vec::new();
        let mut i = 0;
        let bytes = s.as_bytes();
        let flush = |lit: &mut Vec<u8>, pieces: &mut Vec<Piece>| {
            if !lit.is_empty() {
                pieces.push(Piece::Lit(std::mem::take(lit)));
            }
        };
        while i < s.len() {
            if let Some((hole, len)) = parse_hole(s, i) {
                flush(&mut lit, &mut pieces);
                pieces.push(hole);
                i += len;
            } else if collapse_ws && bytes[i].is_ascii_whitespace() {
                flush(&mut lit, &mut pieces);
                while i < s.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                if pieces.last() != Some(&Piece::Ws) {
                    pieces.push(Piece::Ws);
                }
            } else {
                lit.push(bytes[i]);
                i += 1;
            }
        }
        flush(&mut lit, &mut pieces);
        Self { pieces }
    }

    fn holes(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().filter_map(|p| match p {
            Piece::Hole(n) | Piece::Ident(n) => Some(n.as_str()),
            _ => None,
        })
    }
}

pub(crate) struct Rule {
    pattern: Template,
    with: Template,
}

impl Rule {
    pub(crate) fn new(pattern: &str, with: &str) -> Result<Self> {
        let mut pattern = Template::parse(pattern.trim(), true);
        // Surrounding whitespace in a pattern carries no meaning.
        while pattern.pieces.first() == Some(&Piece::Ws) {
            pattern.pieces.remove(0);
        }
        while pattern.pieces.last() == Some(&Piece::Ws) {
            pattern.pieces.pop();
        }
        if !pattern.pieces.iter().any(|p| matches!(p, Piece::Lit(_))) {
            bail!("pattern must contain literal text, not only holes and whitespace");
        }
        if matches!(pattern.pieces.first(), Some(Piece::Hole(_)))
            || matches!(pattern.pieces.last(), Some(Piece::Hole(_)))
        {
            bail!("a pattern cannot start or end with a :[hole]; anchor it with text");
        }
        for pair in pattern.pieces.windows(2) {
            if matches!(pair[0], Piece::Hole(_))
                && matches!(pair[1], Piece::Hole(_) | Piece::Ident(_))
            {
                bail!("adjacent holes in a pattern are ambiguous; separate them with text");
            }
        }
        let with = Template::parse(with, false);
        let defined: Vec<&str> = pattern.holes().collect();
        for name in with.holes() {
            if !defined.contains(&name) {
                bail!("`with` uses hole :[{name}] that the pattern does not define");
            }
        }
        Ok(Self { pattern, with })
    }

    /// Rewrite every non-overlapping match, left to right. Returns the new text
    /// and the number of matches.
    pub(crate) fn apply(&self, text: &str, lang: Lang) -> (String, usize) {
        let src = text.as_bytes();
        let lx = lex(src, lang);
        let mut out = Vec::with_capacity(src.len());
        let mut last = 0;
        let mut count = 0;
        let mut p = 0;
        while p <= src.len() {
            if let Some((end, binds)) = self.match_at(src, &lx, p) {
                let mut rendered = Vec::new();
                self.render(&binds, src, &mut rendered);
                // Deleting everything on a line deletes the line.
                let (from, to) = if rendered.is_empty() {
                    whole_lines(src, p, end).unwrap_or((p, end))
                } else {
                    (p, end)
                };
                out.extend_from_slice(&src[last..from]);
                out.extend_from_slice(&rendered);
                last = to;
                count += 1;
                p = if end > p { end } else { p + 1 };
            } else {
                p += 1;
            }
        }
        out.extend_from_slice(&src[last..]);
        (
            String::from_utf8(out).expect("rewrite preserves UTF-8"),
            count,
        )
    }

    fn render(&self, binds: &Binds, src: &[u8], out: &mut Vec<u8>) {
        for piece in &self.with.pieces {
            match piece {
                Piece::Lit(l) => out.extend_from_slice(l),
                Piece::Ws => out.push(b' '),
                Piece::Hole(n) | Piece::Ident(n) => {
                    let (s, e) = binds[n];
                    out.extend_from_slice(&src[s..e]);
                }
            }
        }
    }

    fn match_at(&self, src: &[u8], lx: &Lexed, start: usize) -> Option<(usize, Binds)> {
        // A match starts in code or exactly at the start of a string/comment.
        if inside_atom(lx, start) {
            return None;
        }
        if let Some(Piece::Lit(l)) = self.pattern.pieces.first()
            && is_ident(l[0])
            && start > 0
            && is_ident(src[start - 1])
        {
            return None;
        }
        let mut binds = BTreeMap::new();
        let end = self.match_from(src, lx, 0, start, &mut binds)?;
        Some((end, binds))
    }

    fn match_from(
        &self,
        src: &[u8],
        lx: &Lexed,
        k: usize,
        p: usize,
        binds: &mut Binds,
    ) -> Option<usize> {
        let Some(piece) = self.pattern.pieces.get(k) else {
            // The match ends in code or exactly at the end of a string/comment,
            // and not in the middle of an identifier.
            if inside_atom(lx, p) {
                return None;
            }
            if let Some(Piece::Lit(l)) = self.pattern.pieces.last()
                && is_ident(*l.last().unwrap())
                && p < src.len()
                && is_ident(src[p])
            {
                return None;
            }
            return Some(p);
        };
        match piece {
            Piece::Lit(l) => {
                if src[p..].starts_with(l) {
                    self.match_from(src, lx, k + 1, p + l.len(), binds)
                } else {
                    None
                }
            }
            Piece::Ws => {
                let mut q = p;
                while q < src.len() && src[q].is_ascii_whitespace() {
                    q += 1;
                }
                self.match_from(src, lx, k + 1, q, binds)
            }
            Piece::Ident(name) => {
                let mut q = p;
                while q < src.len() && is_ident(src[q]) {
                    q += 1;
                }
                if q == p || (k == 0 && p > 0 && is_ident(src[p - 1])) {
                    return None;
                }
                // The identifier is taken whole, unless the pattern continues
                // it with literal text (`:[[stem]]_cuda`): then the longest
                // split that lets the rest match wins.
                let continues = matches!(
                    self.pattern.pieces.get(k + 1),
                    Some(Piece::Lit(l)) if is_ident(l[0])
                );
                let shortest = if continues { p + 1 } else { q };
                (shortest..=q)
                    .rev()
                    .find_map(|e| self.bind(src, lx, k, name, p, e, binds))
            }
            Piece::Hole(name) => {
                if let Some(&(s, e)) = binds.get(name) {
                    // A repeated hole must match the same text again.
                    let want = &src[s..e];
                    return if src[p..].starts_with(want) {
                        self.match_from(src, lx, k + 1, p + want.len(), binds)
                    } else {
                        None
                    };
                }
                // Lazily extend over balanced units.
                let limit = lx
                    .atom_at
                    .get(p)
                    .copied()
                    .flatten()
                    .filter(|&a| lx.atoms[a].start < p);
                let mut q = p;
                loop {
                    if let Some(end) = self.bind(src, lx, k, name, p, q, binds) {
                        return Some(end);
                    }
                    q = next_unit(src, lx, q, limit)?;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(
        &self,
        src: &[u8],
        lx: &Lexed,
        k: usize,
        name: &str,
        s: usize,
        e: usize,
        binds: &mut Binds,
    ) -> Option<usize> {
        if let Some(&(bs, be)) = binds.get(name) {
            if src[bs..be] != src[s..e] {
                return None;
            }
            return self.match_from(src, lx, k + 1, e, binds);
        }
        binds.insert(name.to_string(), (s, e));
        let r = self.match_from(src, lx, k + 1, e, binds);
        if r.is_none() {
            binds.remove(name);
        }
        r
    }
}

// The span of the full lines holding [start, end), including the final newline,
// when nothing but whitespace shares those lines with the match.
fn whole_lines(src: &[u8], start: usize, end: usize) -> Option<(usize, usize)> {
    let line_start = src[..start]
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |i| i + 1);
    if !src[line_start..start]
        .iter()
        .all(|&b| b == b' ' || b == b'\t')
    {
        return None;
    }
    let rest = &src[end..];
    let ws = rest
        .iter()
        .take_while(|&&b| b == b' ' || b == b'\t' || b == b'\r')
        .count();
    match rest.get(ws) {
        Some(b'\n') => Some((line_start, end + ws + 1)),
        None => Some((line_start, src.len())),
        Some(_) => None,
    }
}

fn inside_atom(lx: &Lexed, p: usize) -> bool {
    lx.atom_at
        .get(p)
        .copied()
        .flatten()
        .is_some_and(|a| lx.atoms[a].start < p)
}

// The position after one balanced unit starting at `q`, or None when the hole
// cannot grow: at the end of input, at an unmatched closing bracket, or (for a
// hole that began inside a string) at the closing delimiter.
fn next_unit(src: &[u8], lx: &Lexed, q: usize, limit: Option<usize>) -> Option<usize> {
    if let Some(a) = limit {
        let atom = lx.atoms[a];
        return (q < atom.body_end).then_some(q + 1);
    }
    if q >= src.len() {
        return None;
    }
    if let Some(a) = lx.atom_at[q] {
        return Some(lx.atoms[a].end);
    }
    match src[q] {
        b'(' | b'[' | b'{' => lx.close_of.get(&q).copied(),
        b')' | b']' | b'}' => None,
        _ => Some(q + 1),
    }
}

/// Attribute accesses through `name` in code (`name.attr`): whole word,
/// outside strings and comments, and not itself an attribute (`x.name.attr`).
pub(crate) fn attribute_uses(text: &str, lang: Lang, name: &str) -> usize {
    let src = text.as_bytes();
    let lx = lex(src, lang);
    let pat = name.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while let Some(off) = src[i..].windows(pat.len()).position(|w| w == pat) {
        let s = i + off;
        let e = s + pat.len();
        let before_ok = s == 0 || !(is_ident(src[s - 1]) || src[s - 1] == b'.');
        let rest = &src[e..];
        let ws = rest
            .iter()
            .take_while(|b| **b == b' ' || **b == b'\t')
            .count();
        if before_ok && rest.get(ws) == Some(&b'.') && lx.atom_at[s].is_none() {
            n += 1;
        }
        i = s + 1;
    }
    n
}

// Whole-word occurrences of `name` in code that are not themselves attributes
// (`x.name`), as (start, end) byte ranges.
fn free_occurrences(src: &[u8], lx: &Lexed, name: &str) -> Vec<(usize, usize)> {
    let pat = name.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(off) = src[i..].windows(pat.len()).position(|w| w == pat) {
        let s = i + off;
        let e = s + pat.len();
        let before_ok = s == 0 || !(is_ident(src[s - 1]) || src[s - 1] == b'.');
        let after_ok = e == src.len() || !is_ident(src[e]);
        if before_ok && after_ok && lx.atom_at[s].is_none() {
            out.push((s, e));
        }
        i = s + 1;
    }
    out
}

/// Uses of `name` as a free name in code: whole word, outside strings and
/// comments, and not an attribute of something else.
pub(crate) fn name_uses(text: &str, lang: Lang, name: &str) -> usize {
    let src = text.as_bytes();
    free_occurrences(src, &lex(src, lang), name).len()
}

/// Replace attribute accesses through the dotted `chain` (`a.b.c.attr`) with
/// accesses through `with` (`with.attr`). `chain` must start as a free name
/// and be followed by an attribute access; whitespace inside it is not
/// matched. Returns the new text and the number of replacements.
pub(crate) fn rewrite_attribute_chain(
    text: &str,
    lang: Lang,
    chain: &str,
    with: &str,
) -> (String, usize) {
    let src = text.as_bytes();
    let lx = lex(src, lang);
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    let mut n = 0;
    for (s, _) in free_occurrences(src, &lx, chain.split('.').next().unwrap_or(chain)) {
        let e = s + chain.len();
        if s < last || !text[s..].starts_with(chain) || (e < src.len() && is_ident(src[e])) {
            continue;
        }
        let ws = src[e..]
            .iter()
            .take_while(|b| **b == b' ' || **b == b'\t')
            .count();
        if src.get(e + ws) != Some(&b'.') {
            continue;
        }
        out.push_str(&text[last..s]);
        out.push_str(with);
        last = e;
        n += 1;
    }
    out.push_str(&text[last..]);
    (out, n)
}

/// The identifiers used as free names in code: outside strings and comments,
/// and not attributes of something else.
pub(crate) fn free_names(text: &str, lang: Lang) -> std::collections::BTreeSet<String> {
    let src = text.as_bytes();
    let lx = lex(src, lang);
    let mut out = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < src.len() {
        if lx.atom_at[i].is_some() || !is_ident(src[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < src.len() && is_ident(src[i]) {
            i += 1;
        }
        if !src[s].is_ascii_digit() && (s == 0 || src[s - 1] != b'.') {
            out.insert(text[s..i].to_string());
        }
    }
    out
}

/// A C-family function's name and parameter list, with comments dropped and
/// whitespace normalized, and whether it is a definition (a body follows)
/// rather than a declaration.
pub(crate) struct Signature {
    pub name: String,
    pub params: String,
    pub definition: bool,
}

impl Signature {
    /// The parameter types, which is what must agree between a declaration
    /// and its definition: parameter names and default arguments may differ.
    pub fn types(&self) -> Vec<String> {
        split_top_level(&self.params)
            .into_iter()
            .map(|param| {
                let param = split_default(param).trim();
                let start = param
                    .rfind(|c: char| !(c.is_ascii() && is_ident(c as u8)))
                    .map_or(0, |i| i + 1);
                let (ty, name) = param.split_at(start);
                let ty = ty.trim_end();
                if ty.is_empty()
                    || name.is_empty()
                    || name.as_bytes()[0].is_ascii_digit()
                    || C_TYPE_WORDS.contains(&name)
                    || ty.ends_with("::")
                {
                    param.to_string()
                } else {
                    ty.to_string()
                }
            })
            .collect()
    }
}

// Words that end a type, so a parameter ending in one is unnamed.
const C_TYPE_WORDS: &[&str] = &[
    "bool", "char", "short", "int", "long", "float", "double", "void", "signed", "unsigned",
    "const", "volatile", "auto",
];

// Split at commas outside brackets.
fn split_top_level(params: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in params.char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&params[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if !params[start..].trim().is_empty() {
        out.push(&params[start..]);
    }
    out
}

// A parameter without its default argument.
fn split_default(param: &str) -> &str {
    let mut depth = 0i32;
    for (i, c) in param.char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            '=' if depth == 0 => return &param[..i],
            _ => {}
        }
    }
    param
}

const NOT_FUNCTIONS: &[&str] = &[
    "if", "for", "while", "switch", "return", "sizeof", "catch", "decltype", "alignas",
];

/// Function declarations and definitions in C-family `text`: an unqualified
/// identifier preceded by a type (not by `=`, `return`, `.`, `::`, ...) and
/// followed by a parameter list and then `{` or `;`. Calls are skipped.
pub(crate) fn function_signatures(text: &str) -> Vec<Signature> {
    let src = text.as_bytes();
    let lx = lex(src, Lang::CFamily);
    let skip_ws = |mut i: usize| {
        while i < src.len()
            && (src[i].is_ascii_whitespace()
                || lx.atom_at[i].is_some_and(|a| is_comment(src, &lx.atoms[a])))
        {
            i += 1;
        }
        i
    };
    let mut out = Vec::new();
    let mut e = 0;
    while e < src.len() {
        if lx.atom_at[e].is_some() || !is_ident(src[e]) {
            e += 1;
            continue;
        }
        let s = e;
        while e < src.len() && is_ident(src[e]) {
            e += 1;
        }
        let name = &text[s..e];
        if src[s].is_ascii_digit() || (s > 0 && src[s - 1] == b'.') || NOT_FUNCTIONS.contains(&name)
        {
            continue;
        }
        let open = skip_ws(e);
        if src.get(open) != Some(&b'(') {
            continue;
        }
        let Some(&close) = lx.close_of.get(&open) else {
            continue;
        };
        let definition = match src.get(skip_ws(close)) {
            Some(b'{') => true,
            Some(b';') => false,
            _ => continue,
        };
        // A type ends right before the name; a call follows an operator or keyword.
        let before = text[..s].trim_end();
        let typed = before
            .bytes()
            .last()
            .is_some_and(|b| is_ident(b) || matches!(b, b'>' | b'*' | b'&'))
            && !before.ends_with("return")
            && !before.ends_with("::");
        if !typed {
            continue;
        }
        let mut params = String::new();
        let mut i = open + 1;
        while i < close - 1 {
            if let Some(a) = lx.atom_at[i] {
                let atom = lx.atoms[a];
                if is_comment(src, &atom) {
                    params.push(' ');
                } else {
                    params.push_str(&text[atom.start..atom.end]);
                }
                i = atom.end;
                continue;
            }
            params.push(char::from(src[i]));
            i += 1;
        }
        out.push(Signature {
            name: name.to_string(),
            params: normalize_tokens(&params),
            definition,
        });
    }
    out
}

fn is_comment(src: &[u8], atom: &Atom) -> bool {
    src[atom.start..].starts_with(b"//") || src[atom.start..].starts_with(b"/*")
}

// Collapse whitespace, keeping one space only between two identifier
// characters, so `T &x` and `T& x` compare equal.
fn normalize_tokens(s: &str) -> String {
    let mut out = String::new();
    let mut pending_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            pending_space = true;
            continue;
        }
        if pending_space
            && out
                .chars()
                .last()
                .is_some_and(|p| is_ident(p as u8) && p.is_ascii())
            && c.is_ascii()
            && is_ident(c as u8)
        {
            out.push(' ');
        }
        pending_space = false;
        out.push(c);
    }
    out
}
