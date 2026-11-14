# Managing the Kernels cache

`kernels` downloads a kernel build from the Hub the first time you load it.
After that, loads of the same kernel, revision and build variant come from a
local cache.

The kernels cache is a regular Hugging Face Hub cache: kernel repositories live
in the same directory as `models--…` and `datasets--…` repositories, and they
use the same layout (`refs/`, `blobs/`, `snapshots/`). This page describes how
the cache is configured, what is stored in it, and how to use it when the Hub
is not reachable.

## Cache location

`kernels` uses the first of these settings that's set:

| Setting                     | Cache directory                          |
| --------------------------- | ---------------------------------------- |
| `KERNELS_CACHE`             | `$KERNELS_CACHE`                         |
| `HF_HUB_CACHE`              | `$HF_HUB_CACHE`                          |
| `HUGGINGFACE_HUB_CACHE`     | `$HUGGINGFACE_HUB_CACHE`                 |
| `HF_HOME`                   | `$HF_HOME/hub`                           |
| None of the above           | `$XDG_CACHE_HOME/huggingface/hub`, or `~/.cache/huggingface/hub` when `XDG_CACHE_HOME` is not set |

`KERNELS_CACHE` is the only setting that is specific to `kernels`. Set it when
you want kernel builds somewhere other than the cache of your models, for
example on a fast local disk of a GPU machine, on a shared volume, or in a
directory inside a container image:

```bash
export KERNELS_CACHE=/mnt/fast-disk/kernels
```

When `KERNELS_CACHE` is set, it is used only by `kernels`. Other Hugging Face
tools keep using the Hub cache, so kernels stored in `KERNELS_CACHE` are not
visible to them, and `hf cache` commands do not list them.

Use an absolute path: `kernels` does not expand `~` in `KERNELS_CACHE`.

The variable is read on every call, not at import time, so you can change
`os.environ["KERNELS_CACHE"]` between two `get_kernel` calls and the second call
uses the new directory.

`kernels` uses the same token as other Hub downloads, either `HF_TOKEN` or the
token saved by `hf auth login`. A private kernel repository is cached like any
public kernel, so on a shared machine keep the cache directory readable only by
you.

If you use a Hub mirror through `HF_ENDPOINT`, kernels are downloaded from the
mirror. The cache location is unaffected.

## On-disk layout

Kernel repositories use the `kernels--<org>--<name>` prefix. A cache holding
one version of [`kernels-community/activation`](https://huggingface.co/kernels-community/activation)
looks like this:

```text
<cache-dir>/
├── kernels--kernels-community--activation/
│   ├── refs/
│   │   ├── main
│   │   └── v1
│   ├── blobs/
│   │   └── 9f2c1d…            # file contents, addressed by hash
│   └── snapshots/
│       └── 4d1a8e…/           # one directory per downloaded commit
│           └── build/
│               └── torch28-cxx11-cu128-x86_64-linux/
│                   ├── metadata.json
│                   └── activation/
│                       └── __init__.py
└── .verified-kernels/
    ├── digest/v1/
    └── signature/v1/
```

### `snapshots/`

Each downloaded revision gets one snapshot directory, named after its commit.
A snapshot contains only the files that were downloaded. `kernels` downloads
only the build variant that matches your system (`build/<variant>/*`), so a
snapshot usually holds one variant even when the repository has many.

To populate more than one variant, use [`kernels download`](cli-download.md)
with `--all-variants`, which is useful for building a cache that is shared
between machines with different hardware or PyTorch versions.

### `blobs/`

`blobs/` stores file contents, named by their hash. Files in a snapshot are
symlinks into `blobs/`, or copies on systems without symlinks. Two revisions
that share a file store it only once.

### `refs/`

`refs/` maps a branch or tag, such as `v1`, to the commit it points at.
`kernels` downloads each kernel by commit, so it writes the ref itself when you
load a kernel by version or revision. That's what lets
`get_kernel(..., version=1)` work offline later.

You can delete refs. `kernels` writes them again the next time it downloads the
kernel online. Until then, that branch or tag can't be resolved offline.

### `.verified-kernels/`

`.verified-kernels/` holds verification receipts. Once a kernel passes
verification, `kernels` writes a receipt keyed by repo ID, commit and build
variant, so later loads skip hashing its files and skip verifying the signature
again. Deleting the receipts is safe: the kernel is simply verified in full on
its next load. See [Kernels security](security.md) for what is verified and what
the receipts attest.

## Inspecting the cache

[`kernels cache ls`](cli-cache.md) lists the cached kernel versions with their
build variants, sizes and access times:

```bash
kernels cache ls
```

```text
ID                                  SIZE LAST_ACCESSED  LAST_MODIFIED  VERSION VARIANTS
----------------------------------- ---- -------------- -------------- ------- --------------------------------
kernel/kernels-community/activation 2.5M 45 minutes ago 45 minutes ago v1      torch28-cxx11-cu126-x86_64-linux
                                                                               torch28-cxx11-cu128-x86_64-linux
```

Use `--json` for machine-readable output, `--cache-dir` to inspect a cache other
than the one currently configured, and `--no-truncate` to see full values when
the table is shortened to fit your terminal. If the cache contains
inconsistencies, the command prints a short summary; `--show-warnings` shows the
details. The command works on the local cache only and does not download or load
kernel code.

## Offline use

Set `HF_HUB_OFFLINE=1` to make `kernels` resolve everything from the local
cache, without Hub requests. `HF_HUB_OFFLINE` is read when `huggingface_hub` is
imported, so set it before you start Python.

```bash
export HF_HUB_OFFLINE=1
```

In offline mode, `kernels` looks up versions and revisions in `refs/` and picks
the build variant from the local snapshot. The kernel must already be in the
cache. To add it to the cache, load the kernel once while online:

```python
from kernels import get_kernel

get_kernel("kernels-community/activation", version=1)
```

This writes the `v1` ref as well, so that `version=1` can be resolved offline
later. If your project [locks its kernels](locking.md), run
[`kernels download`](cli-download.md) instead:

```bash
kernels download .
```

A locked project needs no refs offline, since the commit comes from
`kernels.lock`. Use [`--all-variants`](cli-download.md) if the cache also needs
to serve machines with a different GPU or PyTorch version.

If the kernel is not cached, loading fails with an error. For a version:

```text
Version 1 of 'kernels-community/activation' is not available in the local cache
and Hugging Face Hub is in offline mode. Download the kernel while online first,
or pass an explicit `revision=<commit>`.
```

and for a branch or tag:

```text
Cannot resolve revision 'v1' of 'kernels-community/activation' to a commit: the
ref is not in the local cache and Hugging Face Hub is in offline mode. Download
the kernel while online first, or pass an explicit `revision=<commit>`.
```

Passing an explicit commit with `revision=<commit>`, or loading a locked kernel,
needs no ref lookup, so it works offline as long as that commit's snapshot is
cached.

> [!NOTE]
> The publisher trust check cannot be performed offline. When loading from the
> cache in offline mode, `kernels` skips the check and logs a warning. Trust was
> established when the kernel was downloaded.

### Download kernels into a container image

To ship kernels in a container image, download them during the build and set
`KERNELS_CACHE` (or `HF_HOME`) at the resulting directory at runtime:

```dockerfile
ENV KERNELS_CACHE=/opt/kernels-cache
RUN python -c 'from kernels import get_kernel; get_kernel("kernels-community/activation", version=1)'
ENV HF_HUB_OFFLINE=1
```

To read from the cache for a single call without setting `HF_HUB_OFFLINE`, pass
`local_files_only=True` to `install_kernel`:

```python
from kernels import install_kernel

variant_path = install_kernel("kernels-community/activation", version=1, local_files_only=True)
```

## Cleaning up

- Delete a `kernels--…` directory to free space or force a fresh download.
  `kernels` downloads the kernel again the next time you load it.
- Removing `.verified-kernels/` only costs re-verification on the next load.
- When copying or moving a cache, preserve symlinks (`cp -a`, `tar`, `rsync -a`)
  to keep blobs deduplicated. Copying with symlinks resolved also works, but
  stores every file once per snapshot.
- Don't edit cached files. A modified file fails digest verification the next
  time the kernel is verified in full.

## See also

- [Environment variables](env.md) - `KERNELS_CACHE`, `HF_HUB_OFFLINE` and other settings
- [`kernels cache ls`](cli-cache.md) - List cached kernels, versions and variants
- [`kernels download`](cli-download.md) - Pre-populate the cache for locked kernels
- [Lock kernel versions](locking.md) - Pin commits so no version lookup is needed
- [Kernels security](security.md) - Verification and receipts
- [Hugging Face Hub cache reference](https://huggingface.co/docs/huggingface_hub/package_reference/cache) - The cache structures that kernels reuse
