# kernels cache ls

List kernels stored in the local Hugging Face cache. The output includes the
same repository details as `hf cache ls`, plus all cached build variants.
The command runs locally and does not download or load kernel code.

## Usage

```bash
kernels cache ls [--cache-dir PATH] [--json]
```

| Option | Description |
| --- | --- |
| `--cache-dir PATH` | Inspect a specific cache directory. |
| `--json` | Print cached kernel information as JSON. |

By default, the command uses the same cache as `get_kernel`: `KERNELS_CACHE`
when set, otherwise the Hugging Face Hub cache configured by `HF_HUB_CACHE`
or `HF_HOME`. An explicit `--cache-dir` takes precedence over these settings.

## Output

```text
ID                                  SIZE LAST_ACCESSED  LAST_MODIFIED  REFS VARIANTS
----------------------------------- ---- -------------- -------------- ---- --------------------------------
kernel/kernels-community/activation 2.5M 45 minutes ago 45 minutes ago v1   torch28-cxx11-cu128-x86_64-linux
```

`VARIANTS` contains a sorted, comma-separated list of all build variants present
in the repository's cached snapshots. Repeated variant names appear once.
Variants are included regardless of compatibility with the current environment.
Only recognized build directories with cached files are listed; empty build
directories are excluded. If there are no cached variants, the column shows `-`.

## JSON output

```bash
kernels cache ls --json
```

The output contains `cache_dir` and a `repos` list. Each repository contains:

- `repo_id`: the kernel repository ID.
- `size_on_disk`: the repository's cached size in bytes.
- `last_accessed`: the repository's last access time as a Unix timestamp.
- `last_modified`: the repository's last modification time as a Unix timestamp.
- `refs`: all cached refs, or an empty list.
- `variants`: all cached build variant names, without duplicates.

## See also

- [kernels variants](cli-variants.md) - Inspect build variants available on the Hub
- [kernels download](cli-download.md) - Download locked kernels
- [Environment variables](env.md) - Configure cache locations
