# kernels variants

Use `kernels variants` to list the build variants of the latest kernel version
on the Hub, together with compatibility decisions for your current system.
Accepted variants are marked as compatible, the preferred variant is identified,
and rejected variants include the reason they cannot be used.

## Usage

```bash
kernels variants <repo_id> [--all-versions | --version VERSION | --revision REVISION] [--only-compatible]
```

| Option | Description |
| --- | --- |
| `--all-versions` | Show variants and decisions for every available version, in ascending version order. |
| `--only-compatible` | Only show variants compatible with the current system, including the preferred variant. Can be combined with any version or revision selector. |
| `--version VERSION` | Show variants for a specific integer kernel version, including version `0`, or use `latest` for the highest available version. |
| `--revision REVISION` | Show variants for a specific branch, tag, or commit, including repositories without numbered versions. |

`--all-versions`, `--version`, and `--revision` are mutually exclusive.
Without a selector, the command uses the highest available version number; it
does not fall back to an older version when no variants are compatible.
If the repository has no numbered versions, use `--revision` to select a revision.

## Examples

List variants and compatibility decisions for the latest version:

```bash
kernels variants kernels-community/activation
```

You can also select the latest version explicitly:

```bash
kernels variants kernels-community/activation --version latest
```

List variants for every version (the previous `kernels versions` behavior):

```bash
kernels variants kernels-community/activation --all-versions
```

Show only compatible variants for version 1:

```bash
kernels variants kernels-community/activation --version 1 --only-compatible
```

Inspect variants on a branch:

```bash
kernels variants kernels-community/activation --revision main
```

## Example Output

The variants and decisions depend on the repository and your current system.
For example, on a Linux CPU system with Torch 2.10:

```text
Version 1:

torch210-cpu-x86_64-linux compatible, preferred ✅
torch-cpu compatible ✅
```

With `--only-compatible`, the command prints `No compatible variants found.`
when none match the current system.

## Migration from kernels versions

`kernels versions <repo_id>` is deprecated and will be removed in kernels 0.20.
It continues to list all versions
and prints a deprecation message to stderr. Replace it with
`kernels variants <repo_id> --all-versions` to preserve that behavior.

## See Also

- [kernels info](cli-info.md) - Describe a kernel
- [kernels lock](cli-lock.md) - Lock kernel versions in your project
- [kernels download](cli-download.md) - Download locked kernels
