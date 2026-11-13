# Environment variables

## `KERNELS_CACHE`

The directory to use as the local kernel cache. If not set, the cache
of the `huggingface_hub` package is used.

## `UNSTABLE_KERNELS_ALL_LATEST`

Set to `1` to ignore requested kernel versions, revisions, and lockfile pins
when loading kernels. This is an **unstable development mode**: kernel API
changes may break your code. Do not enable it in production.

```bash
UNSTABLE_KERNELS_ALL_LATEST=1 python your_script.py
```

This applies to `get_kernel`, `has_kernel`, `get_locked_kernel`, `load_kernel`,
and the Hub dependencies of `get_local_kernel`, including transitive dependencies.
Layer and function mappings, including locked repositories, follow the same policy.
The highest numbered version branch (for example, `v10` rather than `v2`) is
selected. Repositories without numbered versions raise an error; this mode never
falls back to `main`. If the selected version has no compatible build, loading
fails instead of falling back to an older version. A warning reports the
requested version or revision and the selected ref and commit.

Even with this mode enabled, `get_kernel` and `has_kernel` require an explicit
`version` or `revision`. Passing both remains an error. Local paths and
`LOCAL_KERNELS` overrides take precedence where supported. Lockfile-based loaders
still require a lockfile containing the requested kernel, but ignore its pins
and resolve dependencies from the selected version's metadata. Lockfiles are
not modified. Installation and lockfile generation commands are unaffected.

In offline mode, only cached numbered version refs are considered. If none are
available, loading raises an error even if `main` is cached. `load_kernel` always
uses the cache, even when online. The selected version's build must already be
cached; this may differ from the latest version on the Hub.

`ON`, `YES`, and `TRUE` also enable this mode (case-insensitive). Unsetting the
variable, or any other value including `0`, restores normal pinning. The variable
is read on each loader call.

## `DISABLE_KERNEL_MAPPING`

Disables kernel mappings for [`layers`](layers.md).

## `HF_HUB_DISABLE_TELEMETRY`

Disable telemetry.
