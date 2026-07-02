# kernels info

Use `kernels info` to describe a kernel from a Hub repo ID or a local path.
It prints the kernel metadata (name, version, license, upstream/source
repositories, Python dependencies, and supported backends) and lists each
build variant along with its compatibility with the current system.

## Usage

```bash
kernels info <repo_id_or_path>
```

When the argument is an existing local directory, it is treated as a kernel
repository with build variants in its `build/` directory. Otherwise it is
treated as a Hub repo ID.

Options:

- `--revision <revision>`: describe a specific branch, tag, or commit
  (Hub only). Cannot be used together with `--version`.
- `--version <version>`: describe a specific kernel version (Hub only).
  Cannot be used together with `--revision`. When neither is given, the
  latest version is used (or `main` for unversioned repositories).
- `--json`: print the kernel information as JSON for machine-readable output.

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

## Example Output

```text
Repository: kernels-community/activation
Revision: v1
Name: activation
Version: 1
License: Apache-2.0
Upstream: -
Source: -
Python dependencies: -
Backends: cuda, metal
Variants:

torch212-metal-aarch64-darwin compatible, preferred ✅
torch211-metal-aarch64-darwin: Torch version (2.11) does not match environment Torch version (2.12)
torch211-cxx11-cu128-x86_64-linux: CPU (x86_64) does not match system CPU (aarch64)
torch210-cxx11-cu126-aarch64-linux: OS (linux) does not match system OS (darwin)
...
```

## See Also

- [kernels versions](cli-versions.md) - List available versions of a kernel
- [kernels download](cli-download.md) - Download locked kernels
