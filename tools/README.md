# Kernel conversion tools

Small, single-purpose utilities for helping to convert an upstream project 
into a `kernel-builder` kernel. They are meant to be driven by a coding agent, 
so each one does exactly one thing, is safe to run repeatedly, and reports
machine-readable results. This helps in ensuring deterministic outputs.

These "tools" can be run like so:

```bash
python tools/<tool>.py --help
```

## Conventions

The two **fetchers** (`latest_tag.py`, `fetch_upstream.py`) read from a
remote. The two **rewriters** (`relativize_imports.py`,
`fix_op_definitions.py`) edit files in place. All four share the same
contract:

| | |
| --- | --- |
| Dry run by default | The rewriters change nothing unless `--write` is passed. |
| `--json` | Prints a single JSON object on stdout and nothing else. Human progress output goes to stderr, so the output is always safe to pipe. |
| `--diff` | Prints a unified diff of what changed (or would change). |
| Exit `0` | Clean: nothing to do, or everything was applied. |
| Exit `1` | Work is pending (dry run found changes), or something needs a human. |
| Exit `2` | The tool could not run (bad arguments, unreadable path, `git` failure). |

### Changes vs. issues

A **change** is a rewrite the tool can prove correct on its own. An
**issue** is a problem it found but cannot fix without information from
outside the file, so it reports it instead of guessing.

Guessing would be worse than doing nothing. For example, in `flash-attn-ops`, 
a wrapper calls `custom_op(name, ...)` with `name` as a parameter — and its
callers already pass `add_op_namespace_prefix("...")`. "Fixing" that
call would double-prefix the op into `ns::ns::...`, which builds fine
and breaks later.

Issues keep the exit code at `1` even after `--write`. So exit `0` after
`--write` means the tree is genuinely done, and exit `1` means read the
`issues` list.

With `--json`, both rewriters emit this report. The fetchers emit their own,
simpler objects, shown with each tool below.

```json
{
  "tool": "fix-op-definitions",
  "mode": "check",
  "ok": false,
  "summary": {"files_scanned": 3, "files_changed": 1, "changes": 2, "issues": 0},
  "changed_files": ["torch-ext/relu/op.py"],
  "changes": [
    {"file": "torch-ext/relu/op.py", "line": 5, "kind": "op-name",
     "before": "\"relu::relu_fwd\"", "after": "add_op_namespace_prefix(\"relu_fwd\")"},
    {"file": "torch-ext/relu/op.py", "line": 2, "kind": "import",
     "before": "", "after": "from ._ops import add_op_namespace_prefix"}
  ],
  "issues": []
}
```

Every edit is its own entry, tagged by `kind` — rewriting an op name and
adding the `_ops` import it needs are two changes, not one.

## `latest_tag.py` — resolve the newest release tag

Lists tags with `git ls-remote`, so nothing is cloned.

```bash
python tools/latest_tag.py https://github.com/Dao-AILab/flash-attention.git
# v2.8.3.post1
```

Versions are extracted from the tag name, which handles prefixed tags:
`fa4-v4.0.0.beta23` parses as `4.0.0.beta23`, not `4`. Ordering follows PEP
440, so `beta23` sorts above `beta9` and `v1.2.3.post1` above `v1.2.3`.

Pre-releases are skipped unless `--include-prerelease`, so the default
answer is the latest *stable* tag. `--tag-pattern REGEX` restricts which
tags are considered, which matters for repositories that ship several
product lines from one tag namespace:

```bash
python tools/latest_tag.py https://github.com/Dao-AILab/flash-attention.git \
    --tag-pattern '^fa4-' --include-prerelease --json
```

With `--json` the output also carries the tag's commit SHA and the next
`--limit` candidates.

## `fetch_upstream.py` — clone a repository at a tag

```bash
python tools/fetch_upstream.py https://github.com/Dao-AILab/flash-attention.git upstream/
python tools/fetch_upstream.py https://github.com/Dao-AILab/flash-attention.git upstream/ \
    --tag v2.8.3 --strip-git --json
```

`--tag latest` (the default) resolves through `latest_tag.py` and accepts
the same `--tag-pattern` / `--include-prerelease` flags. Any other value is
verified against the remote before cloning, so a typo fails immediately
instead of producing an empty checkout.

The clone is shallow and pinned to the tag. The resolved commit SHA is
reported, so it can be recorded in the sync commit or PR description.
`--strip-git` drops the `.git` directory for vendoring, and `--force`
replaces a non-empty destination (without it, a non-empty destination is an
error).

## `relativize_imports.py` — make intra-kernel imports relative

Hub kernels are loaded from a build variant directory whose name is *not*
the package name, so an absolute import of the kernel's own modules breaks
at load time. Every intra-kernel import must be relative — see
[kernel requirements](../docs/source/kernel-requirements.md).

```bash
python tools/relativize_imports.py torch-ext/flash_attn4          # dry run
python tools/relativize_imports.py torch-ext/flash_attn4 --write
```

For a file at `<root>/a/b/mod.py`:

| Before | After |
| --- | --- |
| `from <pkg>.utils import x` | `from ...utils import x` |
| `from <pkg> import x` | `from ... import x` |
| `import <pkg>.utils as u` | `from ... import utils as u` |
| `import <pkg>.utils.helpers` | `from ...utils import helpers`, and `<pkg>.utils.helpers.` references renamed to `helpers.` |

The last row needs the reference rewrite because `import a.b.c` binds `a`
while the relative form binds `c`; there is no relative spelling that binds
a dotted name.

Which absolute names count as intra-package is decided by:

* **auto-detection** (default): a top-level name that is the package
  directory name, or that exists as a module or subpackage inside it. This
  covers vendored dependencies (`quack/`, `src/`) with no configuration.
* **`--module-map SRC=DST`**, for upstream packages that were renamed while
  copying. `DST` is a package-relative dotted path; an empty `DST` means the
  package root:

  ```bash
  # upstream `liger_kernel.ops.*` was copied to torch-ext/liger_kernels/*
  python tools/relativize_imports.py torch-ext/liger_kernels \
      --module-map liger_kernel.ops= --write
  ```

`--no-auto` restricts rewriting to `--module-map` entries only.

Not auto-fixed, reported for review:

* `unresolved-target` — the import maps to a module that is not in the
  package, usually a file that was not copied.
* `unrepresentable-import` — `import <pkg>` binds the package root, which
  has no relative spelling. Import the submodules actually used instead.
* `dangling-reference` — a reference to a name the rewrite left unbound.
* `compound-import` — a multi-name `import` that shares its line with other
  code, so it cannot be split.

## `fix_op_definitions.py` — namespace Torch op registrations

Several versions of the same kernel can be loaded into one process, so each
kernel gets a unique Torch op namespace at build time. Op names must never
hardcode a namespace; they go through `add_op_namespace_prefix` from the
generated `_ops` module — see
[writing kernels](../docs/source/builder/writing-kernels.md).

```bash
python tools/fix_op_definitions.py torch-ext/hello          # dry run
python tools/fix_op_definitions.py torch-ext/hello --write
```

```python
# before
@torch.library.custom_op("relu::relu_fwd", mutates_args=())

# after
from ._ops import add_op_namespace_prefix

@torch.library.custom_op(add_op_namespace_prefix("relu_fwd"), mutates_args=())
```

A hardcoded namespace (`relu::`) is stripped, since
`add_op_namespace_prefix` supplies the real one. The `from ._ops import
add_op_namespace_prefix` line is added when missing, with the right number
of dots for the file's depth. `_ops.py` itself is skipped.

Only calls reached through the `torch.library` module count. Same-named
methods on the object returned by `custom_op`
(`my_op.register_autograd(backward, ...)`) and on `torch.library.Library`
(`lib.define(...)`) take a function or a bare schema rather than a
qualified op name, and are left alone.

Not auto-fixed, reported for review:

* `non-literal-op-name` — the op name is a variable or expression, so the
  prefix cannot be added automatically.
* `hardcoded-library-namespace` — `torch.library.Library(...)` fixes the
  namespace at construction.
* `rewrapped-helper` / `fallback-import` — the re-wrapping antipatterns that
  `writing-kernels.md` warns about. Both defeat static analysis of ops and
  can silently produce non-unique namespaces.

## Tests

```bash
python -m pytest tools/tests
```
