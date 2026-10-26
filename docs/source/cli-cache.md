# kernels cache ls

List kernel versions stored in the local Hugging Face cache.
The command runs locally and does not download or load kernel code.

## Usage

```bash
kernels cache ls [--cache-dir PATH] [--json] [--no-truncate] [--show-warnings]
```

| Option | Description |
| --- | --- |
| `--cache-dir PATH` | Inspect a specific cache directory. |
| `--json` | Print cached kernel information as JSON. |
| `--no-truncate` | Display full table values regardless of terminal width. |
| `--show-warnings` | Show individual cache inconsistencies instead of a short summary. |

By default, the command uses the same cache as `get_kernel`: `KERNELS_CACHE`
when set, otherwise the Hugging Face Hub cache configured by `HF_HUB_CACHE`
or `HF_HOME`. An explicit `--cache-dir` takes precedence over these settings.

## Output

Example output in a wide terminal:

```text
ID                                  SIZE LAST_ACCESSED  LAST_MODIFIED  VERSION VARIANTS
----------------------------------- ---- -------------- -------------- ------- --------------------------------
kernel/kernels-community/activation 2.5M 45 minutes ago 45 minutes ago v1      torch28-cxx11-cu126-x86_64-linux
                                                                               torch28-cxx11-cu128-x86_64-linux
                                                                               torch28-cxx11-cu129-x86_64-linux
kernel/kernels-community/activation 2.6M 23 minutes ago 23 minutes ago v2      torch28-cxx11-cu126-x86_64-linux
                                                                               torch28-cxx11-cu128-x86_64-linux
                                                                               torch28-cxx11-cu129-x86_64-linux
```

`VERSION` shows version labels such as `v1` and `v2`, in ascending numeric order.
The listing includes only cached `vN` versions. Other branches, tags, and detached
snapshots are omitted. Each version is listed separately, even when multiple
versions point to the same cached snapshot.

`VARIANTS` lists the builds present in that version's cached snapshot, one per line.
Continuation lines leave the other columns blank. A variant cached for several
versions appears under each of those versions.
Variants are included regardless of compatibility with the current environment.
Only recognized build directories with cached files are listed; empty build
directories are excluded. If there are no cached variants, the column shows `-`.

`SIZE`, `LAST_ACCESSED`, and `LAST_MODIFIED` describe the cached snapshot for that
version. See the [Hugging Face Hub cache reference](https://huggingface.co/docs/huggingface_hub/package_reference/cache#data-structures)
for details on sizes and timestamps.

Like `hf cache ls`, the table shortens long values to fit the terminal width.
Use `--no-truncate` or `--json` to see all values in full. Cache inconsistencies
produce a short summary by default; use `--show-warnings` to see their details.

## JSON output

```bash
kernels cache ls --json
```

The output contains `cache_dir` and a `repos` list. Each repository contains:

- `repo_id`: the kernel repository ID.
- `size_on_disk`: the repository's cached size in bytes.
- `last_accessed`: the repository's last access time as a Unix timestamp.
- `last_modified`: the repository's last modification time as a Unix timestamp.
- `versions`: one entry per cached `vN` version. Each entry contains:
  - `version`: the version label, such as `"v1"`.
  - `size_on_disk`, `last_accessed`, and `last_modified`: the size in bytes and
    Unix timestamps for that version's cached snapshot.
  - `variants`: the sorted build variant names in that snapshot.

Repository-level sizes and timestamps describe all its cached snapshots;
version-level values describe just the corresponding snapshot.

## See also

- [kernels variants](cli-variants.md) - Inspect build variants available on the Hub
- [kernels download](cli-download.md) - Download locked kernels
- [Environment variables](env.md) - Configure cache locations
