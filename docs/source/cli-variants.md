# kernels variants

Use `kernels variants` to list the build variants of the latest kernel version
on the Hub, together with compatibility decisions for your current system.
Accepted variants are marked as compatible, the preferred variant is identified,
and rejected variants include the reason they cannot be used.

## Usage

```bash
kernels variants <repo_id> [--all-versions | --version VERSION | --revision REVISION] [--only-compatible] [--json]
```

| Option | Description |
| --- | --- |
| `--all-versions` | Show variants and decisions for every available version, in ascending version order. |
| `--only-compatible` | Only show variants compatible with the current system, including the preferred variant. Can be combined with any version or revision selector. |
| `--version VERSION` | Show variants for a specific integer kernel version, including version `0`, or use `latest` for the highest available version. |
| `--revision REVISION` | Show variants for a specific branch, tag, or commit, including repositories without numbered versions. |
| `--json` | Print build variants and compatibility decisions as JSON for machine-readable output. Can be combined with any selector and `--only-compatible`. |

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

## JSON Output

Get machine-readable output:

```bash
kernels variants kernels-community/activation --version 1 --json
```

The output is a single JSON object with the repository ID and a `revisions` list,
including when selecting just one version or revision:

```json
{
  "repo_id": "kernels-community/activation",
  "revisions": [
    {
      "version": 1,
      "revision": "refs/heads/v1",
      "variants": [
        {
          "variant": "torch210-cpu-x86_64-linux",
          "compatible": true,
          "preferred": true,
          "reason": null
        },
        {
          "variant": "torch-cpu",
          "compatible": true,
          "preferred": false,
          "reason": null
        }
      ]
    }
  ]
}
```

Each variant includes compatibility and preference flags. Rejected variants
include a rejection `reason`; compatible variants have `"reason": null`.
With `--revision`, `version` is `null` and `revision` is the supplied branch, tag,
or commit. With `--all-versions`, entries appear in ascending version order.
When no variants are found, or none remain after `--only-compatible` filtering,
`variants` is an empty list.

## Migration from kernels versions

`kernels versions <repo_id>` is deprecated and will be removed in kernels 0.20.
It continues to list all versions
and prints a deprecation message to stderr. Replace it with
`kernels variants <repo_id> --all-versions` to preserve that behavior.

## See Also

- [kernels info](cli-info.md) - Describe a kernel
- [kernels lock](cli-lock.md) - Lock kernel versions in your project
- [kernels download](cli-download.md) - Download locked kernels
