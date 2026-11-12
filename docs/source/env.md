# Environment variables

## `KERNELS_CACHE`

The directory to use as the local kernel cache. If not set, the cache
of the `huggingface_hub` package is used.

See [Kernels cache](cache.md) for the full lookup order, the cache layout, and
how to inspect or pre-populate the cache.

## `HF_HUB_OFFLINE`

Set to `1` to resolve kernels from the local cache only, without requests to
the Hub. The kernel must already be in the cache. See
[offline use](cache.md#offline-use).

## `DISABLE_KERNEL_MAPPING`

Disables kernel mappings for [`layers`](layers.md).

## `HF_HUB_DISABLE_TELEMETRY`

Disable telemetry.
