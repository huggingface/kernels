# kernels info

Use `kernels info` to describe a kernel from a Hub repo ID or a local path. It
prints the kernel metadata (name, version, license, upstream/source
repositories, and supported backends), along with the recursive kernel and
Python package dependencies of the variant selected for your current system.
Only variant metadata is fetched. Kernel artifacts are not downloaded or
imported, and metadata and kernel validators are not run. To list the build
variants of a kernel and their compatibility with the current system, use
[kernels variants](cli-variants.md).

## Usage

```bash
kernels info <repo_id_or_path>
```

When the argument is an existing local directory, it is treated as a kernel
repository with build variants in its `build/` directory. Otherwise it is
treated as a Hub repo ID.

Options:

- `--revision <revision>`: describe a specific branch, tag, or commit (Hub
  only). Cannot be used together with `--version`.
- `--version <version>`: describe a specific kernel version (Hub only). Cannot
  be used together with `--revision`. When neither is given, the latest version
  is used (or `main` for unversioned repositories).
- `--json`: print the kernel information as JSON for machine-readable output.
- `--dep-tree`: show kernel dependencies as an ASCII tree instead of a flat
  list. With `--json`, include a nested `dependency_tree` object as well as the
  flat lists.

Python dependencies include packages required by the selected kernel and all its
transitive kernel dependencies. Shared kernel dependencies appear once in the
flat list and under each parent in the tree.

The selected variant is shown in the output. If a compatible variant cannot be
found, metadata is unavailable, or a dependency cannot be resolved, the command
reports an error.

## Examples

Describe a kernel on the Hub:

```bash
kernels info kernels-community/activation
```

Describe a local kernel repository (e.g. after `kernel-builder build-and-copy`):

```bash
kernels info ./relu
```

Get machine-readable output:

```bash
kernels info --json kernels-community/activation
```

Show the kernel dependency tree:

```bash
kernels info --dep-tree <repo_id_or_path>
```

JSON output includes `variant`, `kernel_depends` (repository IDs), and
`python_depends` (package requirements). With `--dep-tree --json`,
`dependency_tree` maps each direct dependency's repository ID to its own
dependency tree; leaves are empty objects.

```json
{
  "kernel_depends": ["example/b", "example/c", "example/d"],
  "python_depends": ["numpy", "torch"],
  "dependency_tree": {
    "example/b": { "example/d": {} },
    "example/c": { "example/d": {} }
  }
}
```

## Example Output

```text
Repository: kernels-community/activation
Revision: v1
Variant: torch28-cxx11-cu128-x86_64-linux
Name: activation
Version: 1
Minimum kernels version: -
License: Apache-2.0
Upstream: -
Source: -
Kernel dependencies: -
Python dependencies: -
Backends: cuda, metal
```

For the shared dependency example above, `--dep-tree` displays:

```text
Kernel dependencies:
example/a
|-- example/b
|   `-- example/d
`-- example/c
    `-- example/d
```

## See Also

- [kernels variants](cli-variants.md) - List kernel variants and compatibility
  decisions
- [kernels download](cli-download.md) - Download locked kernels
