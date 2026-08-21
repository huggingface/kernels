# kernel-port

> [!WARNING]
> This is an experiment. The recipe language, the op set, and the CLI are all subject to change without notice, and nothing here is covered by any stability guarantee. Do not depend on it (yet).

Ports a kernel repo into the [kernel-builder](https://github.com/huggingface/kernel-builder) layout by running a recipe.

- A port is a `port.kdl` recipe plus an `overlay/` directory of checked-in files.
- Same pins and same recipe gives byte-identical output, every run.
- Every op fails loudly on drift, so upstream changes cannot silently mis-port.
- Judgment is spent once, while writing the recipe. Running it is mechanical.

## The recipe

A recipe is not a programming language. It is an ordered list of operations, written as a [KDL 2.0](https://kdl.dev) document.

- One node per op, with `key="value"` properties.
- `//` comments, and raw multi-line strings (`#"""..."""#`) for exact-text payloads.
- No variables, no expressions, no control flow. Every pin is a literal.
- KDL features with no meaning here are rejected rather than ignored: positional arguments, children blocks, type annotations.
- A recipe kept on disk opens with a `recipe version=N` header, declaring the format it was written against. See [versioning](#versioning).

## See it work, without a checkout

Two flags remove the setup:

- `-e` takes the recipe inline, instead of a path to one.
- `--file path=content` supplies the input tree, instead of `--dir`.

Nothing is read from disk and nothing is written to it, so one command shows what an op does:

```sh
kernel-port -e 'relativize_imports in="pkg/**" package_root="pkg" changes=1' \
    --file 'pkg/__init__.py=from pkg.ops import hello'
```

```
[line   1] relativize_imports  rewrote 1 import(s) in 1 file(s)
M pkg/__init__.py

FILE: pkg/__init__.py
from .ops import hello
```

Change `changes=1` to `changes=2` and the same command shows the other half of the design. The run refuses to proceed rather than porting something you did not sign off on:

```
error: recipe line 1: relativize_imports: expected exactly 2 change(s) but made 1 - upstream drifted; review the new rewrites and update changes=
```

The [cookbook](#cookbook) is one such command per op.

## Running a real port

```sh
cargo run -p kernel-port -- <recipe>.kdl --dir <upstream-checkout> --out <dir> \
    [--vendor name=dir] [--dry-run] [--diff] [--print] [--partial]
```

- `--dir` is the upstream checkout. It is read, never written.
- `--out` writes the ported tree, wiped and regenerated on every run. Without it, `--dir` is modified in place.
- `--dry-run` computes the changes and writes nothing.
- `--diff` prints unified diffs for changed files, moves included.
- `--print` dumps every file in the resulting tree.
- `--vendor name=dir` supplies a second pinned checkout for [`vendor`](#vendor).
- `--partial` writes the state as of the last successful op to `--out`, to inspect what a failing op saw. The run still exits non-zero.

## Ops

Each op links to its entry in the [cookbook](#cookbook) below, which gives the full argument list, a runnable example, and the failure modes.

| Op | What it does |
| --- | --- |
| [`source`](#source) | Pin the upstream: verify the checkout's HEAD commit, origin URL, and cleanliness |
| [`vendor`](#vendor) | Verify a second pinned upstream (given via `--vendor`) and copy a subtree from it |
| [`prune`](#prune) | Delete everything except the given globs |
| [`delete`](#delete) | Delete files matching a glob (must match something) |
| [`move`](#move) | Rename a file or directory |
| [`overlay`](#overlay) | Copy checked-in files (bindings, flake, docs) over the workspace |
| [`replace`](#replace) | Exact-text find/replace with a required occurrence `count` |
| [`strip_suffix`](#strip_suffix) | Remove one literal suffix from every matched file, with a pinned `files=N` |
| [`expect`](#expect) | Guard: assert an exact text occurs `count` times (0 asserts absence), or that a glob matches `files=N` |
| [`convert_import`](#convert_import) | Rewrite `import a.b.c as x` into `from a.b import c as x` |
| [`remap_module`](#remap_module) | Rewrite `from a.b import x` module prefixes onto a new namespace |
| [`relativize_imports`](#relativize_imports) | Rewrite absolute intra-package imports to minimal-dot relative form |
| [`ensure_init`](#ensure_init) | Add an empty `__init__.py` to any package dir missing one |
| [`kernel`](#kernel) | Record one `[kernel.<name>]` section for the manifest |
| [`manifest`](#manifest) | Generate `build.toml` from the recorded kernel sections (or noarch mode) |

## What the runner enforces

On top of whatever the recipe says:

- `source` and `vendor` require a full 40-character SHA and a clean checkout, untracked files included.
- Every op is built before the first one runs, so an argument typo or a bad glob anywhere fails the recipe before anything is mutated.
- After the last op, every added or modified Python file must still parse.
- No absolute in-package import may remain under `torch-ext`. The Hub loads a kernel under a build-variant directory name, so only relative intra-package imports resolve at run time.
- `--out` runs write a `.port-provenance.json`: recipe hash, runner version, pinned sources, output tree hash. All deterministic, so the file reproduces byte-for-byte.

## Versioning

A recipe is re-run later, by someone else, against a newer build of this tool. The header says which format it was written against.

```kdl
recipe version=1
```

- The header must be the first node in the file. Comments above it are fine.
- A recipe file that does not declare a version is rejected. An inline `-e` recipe may omit it, since it does not outlive the command.
- A version this build does not implement is rejected, rather than run under a meaning the author never saw.
- The version is recorded as `format` in `.port-provenance.json`.

`version` is bumped when a change would give an existing recipe a *different meaning*, not when it gains a new capability. Adding an op or an optional argument leaves every existing recipe alone, so it is not a bump. Changing what an existing argument does is.

## Cookbook

One entry per op: what it takes, what it guarantees, a command you can run as written, and what makes it fail. Ops run top to bottom against an in-memory copy of the tree; nothing reaches disk until every op has succeeded.

Conventions used throughout:

- **Globs** match repo-relative paths with `/` separators. `*` does not cross a `/`, `**` does. `{a,b}` alternates. Args that take several globs (`src`, `keep`, `torch_src`) split on commas outside braces.
- **Pins** (`count=`, `files=`, `changes=`) are literal integers that must match exactly. They are the point: they turn "upstream changed" from a silent mis-port into a failed run.
- **Payload strings** use `"..."` with `\n`/`\t`/`\"` escapes, or KDL's raw multi-line form for exact text that contains quotes and newlines:
  ```kdl
  replace in="a.cpp" count=1 with="" find=#"""
  static auto registry = torch::RegisterOperators()
      .op("pkg::thing", &thing);
  """#
  ```
  The first and last newlines are the delimiters, so the payload above starts at `static` and ends at `;`.
- The examples below build their input with `--file path=content` so they run with no repository at all. `$'...'` is shell syntax for a string containing real newlines. With no `--out`, the run prints the resulting tree; a trailing `...` in an output block means the rest of that dump is elided here.

### The header

#### `recipe`

```kdl
recipe version=1
```

Declare the recipe format, as the first node in the file. Not an op: it runs nothing and is not part of the pipeline. Required in a recipe file, optional inline. See [versioning](#versioning).

Fails when: the version is one this build does not implement, or the header is not first.

### Pinning the upstream

#### `source`

```kdl
source repo="<url>" commit="<40-char sha>"
```

Assert what `--dir` is. Verifies the checkout's `HEAD` is exactly `commit`, that its `origin` URL is `repo`, and that the working tree is clean (untracked files included), then records the pin in `.port-provenance.json`. Every recipe that ports a real repository starts with this line.

```kdl
source repo="https://github.com/rusty1s/pytorch_scatter" commit="f514c10f920b5aeed2eb162092f0ad20d3edee52"
```

Fails when: `--dir` is not a git checkout, sits at a different commit, has a different origin, or has uncommitted or untracked changes.

#### `vendor`

```kdl
vendor name="<id>" repo="<url>" commit="<40-char sha>" path="<subdir>" to="<dir>"
```

The same verification for a second repository, supplied on the command line as `--vendor <id>=<dir>`, then copies its `path` subtree into the workspace at `to`. For kernels that vendor a dependency's sources instead of depending on it.

```kdl
vendor name="quack" repo="https://github.com/Dao-AILab/quack" commit="<sha>" \
    path="quack/cute" to="torch-ext/kernel/cute"
```

Fails when: no `--vendor` was passed for that name, the checkout drifts from the pin, or `path` is not a directory in it.

### Shaping the tree

#### `prune`

```kdl
prune keep="<glob>[,<glob>...]"
```

Delete everything the globs do not match. This is the port's statement of what it carries over, so it belongs near the top: whatever upstream adds later lands outside `keep` and is dropped, rather than silently shipping.

```sh
kernel-port -e 'prune keep="csrc/**,pkg/**"' \
    --file 'csrc/k.cu=// kernel' --file 'pkg/__init__.py=x' --file 'setup.py=setup()'
```

```
[line   1] prune               removed 1 file(s), kept 2
D setup.py
...
```

Fails when: `keep` is empty, or any glob in it matches nothing (a stale keep entry is a bug, not a no-op).

#### `delete`

```kdl
delete in="<glob>"
```

Delete the matching files. Use it for a handful of paths; use `prune` when the list of what to keep is shorter than the list of what to drop.

```sh
kernel-port -e 'delete in="**/*.pyc"' --file 'pkg/a.py=a' --file 'pkg/a.pyc=binary'
```

```
[line   1] delete              removed 1 file(s)
D pkg/a.pyc
...
```

Fails when: the glob matches nothing.

#### `move`

```kdl
move from="<path>" to="<path>"
```

Rename a file, or a whole directory if `from` names one. Moves are recorded, so `--diff` can show a moved file's content change as `old path -> new path` instead of a delete plus an add.

```sh
kernel-port -e 'move from="csrc" to="hello-kernel"' \
    --file 'csrc/k.cu=// kernel' --file 'csrc/k.h=// header'
```

```
[line   1] move                moved 2 file(s) to "hello-kernel"
A hello-kernel/k.cu
A hello-kernel/k.h
D csrc/k.cu
D csrc/k.h
...
```

Fails when: `from` matches no file or directory, or a destination path already exists.

#### `overlay`

```kdl
overlay from="<dir>"
```

Copy a directory of checked-in files over the workspace, overwriting what is there. `from` is relative to the recipe file. This is where the files with no upstream equivalent live: `torch_binding.cpp`, `flake.nix`, a CARD.md. Keep it small - anything derivable from upstream should be an op, not an overlay file, so that upstream drift is detected rather than papered over.

```kdl
overlay from="overlay"
```

Fails when: the directory does not exist or contains no files.

### Editing text

#### `replace`

```kdl
replace in="<glob>" find="<text>" with="<text>" count=N
```

Exact-text find and replace across every matching file, where `count` is the total number of occurrences across all of them. `with=""` deletes the text. No regexes and no capture groups: what you pin is what gets rewritten.

```sh
kernel-port -e 'replace in="*.cpp" find="TORCH_EXTENSION_NAME" with="ops" count=2' \
    --file $'b.cpp=TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {}\nREGISTER(TORCH_EXTENSION_NAME)'
```

```
[line   1] replace             2 replacement(s) in 1 file(s)
M b.cpp

FILE: b.cpp
TORCH_LIBRARY(ops, m) {}
REGISTER(ops)
```

Fails when: the glob matches no files, or the occurrence total is not exactly `count`. The error names the per-file counts it did find.

#### `strip_suffix`

```kdl
strip_suffix in="<glob>" suffix="<text>" files=N
```

Remove one literal suffix from the end of every matching file, with both the number of files and the suffix on each of them pinned. For trailing content that upstream appends uniformly - a generated footer, a license tail.

```sh
kernel-port -e 'strip_suffix in="*.h" suffix="\n// EOF\n" files=1' \
    --file $'k.h=#pragma once\n// EOF\n'
```

```
[line   1] strip_suffix        stripped suffix from 1 file(s)
M k.h
...
```

Fails when: the file count is not `files`, or any matched file does not end with `suffix`.

#### `expect`

```kdl
expect in="<glob>" find="<text>" count=N
expect in="<glob>" files=N
```

A guard that changes nothing. The first form asserts an exact text occurs exactly `count` times across the matched files - `count=0` asserts absence. The second asserts the glob matches exactly `N` files.

Reach for it to state an invariant the rest of the recipe depends on but does not itself enforce: that no absolute import survived, that the upstream source list is still the size you reviewed. Guards hold regardless of which op was supposed to do the work, so they keep holding when the recipe is edited.

```sh
kernel-port -e 'expect in="**/*.cu" files=3' --file 'a.cu=x' --file 'sub/b.cu=y'
```

```
error: recipe line 1: expect: expected "**/*.cu" to match exactly 3 file(s), found 2 - the upstream file set drifted; update the port definition (a.cu, sub/b.cu)
```

Fails when: the count or file count does not match. `find` and `files` are mutually exclusive.

### Rewriting Python imports

These four go through libcst, so comments, quoting, and formatting survive byte-for-byte. Each takes an optional `changes=N` pinning exactly how many import statements it rewrites, which is what stops a newly added upstream file from being rewritten silently.

#### `convert_import`

```kdl
convert_import in="<glob>" prefix="<dotted.path>" [changes=N]
```

Rewrite `import a.b.c as x` into `from a.b import c as x`, for modules under `prefix`. The `from` form is what the later ops can relativize; the `import` form cannot be made relative at all.

```sh
kernel-port -e 'convert_import in="**/*.py" prefix="pkg" changes=1' \
    --file $'m.py=import pkg.ops as ops\nimport os'
```

```
[line   1] convert_import      converted 1 import(s) in 1 file(s)
M m.py

FILE: m.py
from pkg import ops as ops
import os
```

`import os` is untouched: only `prefix` is in scope.

#### `remap_module`

```kdl
remap_module in="<glob>" from="<dotted.path>" to="<dotted.path>" [changes=N]
```

Move a module prefix onto a new namespace. Prefix matching is boundary-aware, so `from="pkg.utils"` does not touch `pkg.utils_extra`.

```sh
kernel-port -e 'remap_module in="**/*.py" from="pkg" to="torch_ext.pkg" changes=2' \
    --file $'m.py=from pkg.ops import a\nfrom pkg.util import b'
```

```
[line   1] remap_module        rewrote 2 import(s) in 1 file(s)
M m.py

FILE: m.py
from torch_ext.pkg.ops import a
from torch_ext.pkg.util import b
```

#### `relativize_imports`

```kdl
relativize_imports in="<glob>" package_root="<dir>" [root_relative=#true] [changes=N]
```

Rewrite absolute intra-package imports to their minimal-dot relative form. This is required, not cosmetic: the Hub loads a kernel under a build-variant directory name, so an absolute self-import resolves to nothing at run time. The runner re-checks this after the last op regardless of whether you ran this op.

`package_root` is the package directory itself (`torch-ext/<pkg>`); each file's own package is derived from where it sits under it. `root_relative=#true` rewrites relative to the package root instead of the file, keeping the full module path visible (`from ...ops import base` rather than `from . import base`).

```sh
kernel-port -e 'relativize_imports in="pkg/**" package_root="pkg" changes=1' \
    --file 'pkg/__init__.py=from pkg.ops import hello'
```

```
[line   1] relativize_imports  rewrote 1 import(s) in 1 file(s)
M pkg/__init__.py

FILE: pkg/__init__.py
from .ops import hello
```

#### `ensure_init`

```kdl
ensure_init under="<dir>" [changes=N]
```

Add an empty `__init__.py` to every directory under `under` that holds Python files but has no `__init__.py`. Upstream layouts that relied on namespace packages or on setuptools discovery need this to import as a package.

```sh
kernel-port -e 'ensure_init under="torch-ext/pkg" changes=1' \
    --file 'torch-ext/pkg/__init__.py=x' --file 'torch-ext/pkg/sub/a.py=y'
```

```
[line   1] ensure_init         added 1 missing __init__.py file(s)
A torch-ext/pkg/sub/__init__.py
...
```

Fails when: nothing exists under `under`.

### Generating the manifest

#### `kernel`

```kdl
kernel name="<id>" backend="<cuda|rocm|cpu|metal|xpu|...>" src="<glob>[,<glob>...]"
    [include="<dir>,..."] [depends="torch,..."] [capabilities="8.0,9.0"]
    [cxx_flags="..."] [cuda_flags="..."] [cuda_minver="12.8"]
    [rocm_archs="gfx942,..."] [repeat_src="<path>,..."]
```

Record one `[kernel.<name>]` section for the manifest. It writes no files; the `manifest` op emits everything recorded before it. One `kernel` per backend, and the `src` globs are resolved when the op runs, so a source file added upstream inside an already-matched directory is picked up (and one added outside it is not - that is what `expect ... files=N` is for).

`depends` defaults to `torch`. `repeat_src` lists paths that must be compiled twice, and each must already be selected by `src`.

#### `manifest`

```kdl
manifest name="<id>" backends="<b>[,<b>...]" torch_src="<glob>[,<glob>...]"
    [version=N] [edition=N] [license="..."] [upstream="..."]
    [repo_id="org/name"] [hub_branch="..."] [python_depends="..."]
    [cuda_minver="..."] [cuda_maxver="..."] [cuda_python_depends="..."]
    [torch_pyext="py,pyi,..."] [torch_include="<dir>,..."]
    [stable_abi="cuda=2.11,..."] [stable_abi_version="2.11"]

manifest name="<id>" backends="..." noarch=#true [noarch_pyext="..."]
```

Generate `build.toml` from the `kernel` sections recorded before it, plus the `[general]` and `[torch]` settings given here. The manifest is always generated, never overlaid: if it needs a field this op cannot emit, extend the op.

`torch_src` selects the binding sources; `torch_include` adds include directories for them (which is how a binding can `#include` a header from the kernel directory instead of restating its declarations). `noarch=#true` switches to a `[torch-noarch]` manifest and rejects the torch-only arguments.

```sh
kernel-port --file 'hello-kernel/k.cu=// kernel' --file 'torch-ext/binding.cpp=// binding' \
    -e $'kernel name="hello" backend="cuda" src="hello-kernel/**" capabilities="8.0,9.0"\nmanifest name="hello" backends="cuda" torch_src="torch-ext/*.cpp" version=1 license="Apache-2.0"'
```

```
[line   1] kernel              declared [kernel.hello] with 1 src file(s)
[line   2] manifest            wrote build.toml (1 torch src, 1 kernel section(s))
A build.toml

FILE: build.toml
[general]
name = "hello"
version = 1
license = "Apache-2.0"
backends = ["cuda"]

[torch]
src = [
    "torch-ext/binding.cpp",
]

[kernel.hello]
backend = "cuda"
cuda-capabilities = ["8.0", "9.0"]
depends = ["torch"]
src = [
    "hello-kernel/k.cu",
]
...
```

Fails when: no `kernel` section was declared (and `noarch` is not set), a glob matches nothing, or an `include` directory holds no files.

### A whole port in one command

The ops composed: an upstream layout goes in, a kernel-builder layout comes out, and still nothing touches the disk.

```sh
kernel-port \
    --file 'hello/__init__.py=from hello.ops import hello' \
    --file $'hello/ops.py=from hello import _C\n\ndef hello(x):\n    return _C.hello(x)' \
    --file 'csrc/k.cu=// kernel' \
    --file 'torch-ext/binding.cpp=// op registration' \
    --file 'setup.py=setup()' \
    -e '
prune keep="csrc/**,hello/**,torch-ext/**"
move from="csrc" to="hello-kernel"
move from="hello" to="torch-ext/hello"
relativize_imports in="torch-ext/hello/**" package_root="torch-ext/hello" changes=2
replace in="torch-ext/hello/ops.py" find="from . import _C" with="from ._ops import ops as _C" count=1
kernel name="hello" backend="cuda" src="hello-kernel/**"
manifest name="hello" backends="cuda" torch_src="torch-ext/*.cpp"'
```

```
[line   2] prune               removed 1 file(s), kept 4
[line   3] move                moved 1 file(s) to "hello-kernel"
[line   4] move                moved 2 file(s) to "torch-ext/hello"
[line   5] relativize_imports  rewrote 2 import(s) in 2 file(s)
[line   6] replace             1 replacement(s) in 1 file(s)
[line   7] kernel              declared [kernel.hello] with 1 src file(s)
[line   8] manifest            wrote build.toml (1 torch src, 1 kernel section(s))
...

FILE: torch-ext/hello/__init__.py
from .ops import hello

FILE: torch-ext/hello/ops.py
from ._ops import ops as _C

def hello(x):
    return _C.hello(x)
```

Drop the `relativize_imports` and `replace` lines and every op still succeeds - but the run fails anyway, because the verify stage runs after the last op:

```
error: verify: absolute in-package imports remain under torch-ext (they must be relative):
  torch-ext/hello/__init__.py: from hello.ops import ...
  torch-ext/hello/ops.py: from hello import ...
```
