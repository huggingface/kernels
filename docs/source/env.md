# Environment variables

This document describes the environment variables that configure `kernels`, including cache directory resolution, offline operation, authentication, and telemetry.

## `KERNELS_CACHE`

The directory to use as the local kernel cache. When set, this overrides the default Hugging Face Hub cache directory specifically for `kernels`.

```bash
export KERNELS_CACHE=/path/to/custom/kernel_cache
```

Leading tildes (`~`) are expanded to the user's home directory (for example, `~/my_kernels_cache`).

### Cache directory precedence

When resolving where kernels and receipts are stored on disk, `kernels` checks the following locations in order of precedence:

1. **`KERNELS_CACHE`**: Dedicated cache directory for kernels.
2. **`HF_HUB_CACHE`** or **`HUGGINGFACE_HUB_CACHE`**: Standard Hugging Face Hub cache directory.
3. **`$HF_HOME/hub`**: Hub subfolder inside the Hugging Face home directory.
4. **`$XDG_CACHE_HOME/huggingface/hub`**: XDG cache directory fallback.
5. **`~/.cache/huggingface/hub`**: Default user home cache directory.

If none of these variables are set and the user's home directory cannot be determined, an `UnknownCacheDir` error is raised.

### Cache structure

The cache directory stores downloaded kernels, version tags, and verification state using the standard Hub repository layout:

* **Snapshots (`snapshots/<commit_sha>/`)**: Contains immutable files for a specific kernel commit, including compiled shared objects (`.so`/`.dylib`/`.dll`), Python interfaces, and build metadata (`build.toml`).
* **Refs (`refs/<version_or_branch>`)**: Stores the resolved commit SHA for a version (such as `v1`). When a kernel version is resolved, `kernels` records this mapping in the cache so subsequent loads can resolve offline without querying the Hub.
* **Signature verification receipts (`receipts/`)**: To avoid computing SHA-256 digests over all kernel files on every load, `kernels` writes a verification receipt upon the first successful signature check. Receipts are content-addressed by repository ID, commit SHA, and build variant. Subsequent loads verify the certificate policy against this receipt, skipping expensive file re-hashing while maintaining security.

## `HF_HUB_OFFLINE`

Enables offline mode across `kernels`. When set to a truthy value (`1`, `ON`, `YES`, or `TRUE`), all network calls to the Hugging Face Hub are disabled, and `kernels` will only load from the local cache.

```bash
export HF_HUB_OFFLINE=1
```

This is equivalent to passing `local_files_only=True` to functions such as [`~kernels.get_kernel`], [`~kernels.has_kernel`], or [`~kernels.install_kernel`].

> [!TIP]
> To use `kernels` in an air-gapped or offline environment, pre-download the required kernels on a machine with internet access (or using [`kernels download`](cli-download.md)), and copy the cache directory to the target environment.

## `HF_HOME`

Specifies the root directory for all Hugging Face data (defaults to `~/.cache/huggingface`). If `KERNELS_CACHE` and `HF_HUB_CACHE` are unset, kernels will be stored under `$HF_HOME/hub`.

```bash
export HF_HOME=/data/huggingface
```

## `HF_HUB_CACHE`

Specifies the cache directory for Hugging Face Hub downloads (defaults to `$HF_HOME/hub`).

## `HF_TOKEN`

The authentication token used to access gated or private kernel repositories. `kernels` resolves authentication tokens in the following order:

1. Explicit `token` argument passed to functions or CLI commands.
2. The `HF_TOKEN` environment variable (or `HUGGING_FACE_HUB_TOKEN`).
3. Stored token file at `HF_TOKEN_PATH` (or `$HF_HOME/token`).

Implicit token resolution from the environment or token file can be disabled by setting `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`.

## `HF_ENDPOINT`

Overrides the default Hugging Face Hub endpoint (defaults to `https://huggingface.co`). Useful when operating behind an internal mirror or proxy.

```bash
export HF_ENDPOINT=https://my-internal-hub-mirror.corp
```

## `DISABLE_KERNEL_MAPPING`

Disables kernel mappings for [`layers`](layers.md). When set, layers will not dispatch to kernel implementations.

## `HF_HUB_DISABLE_TELEMETRY`

Disables telemetry collection in the `User-Agent` header sent during Hub API calls. By default, telemetry includes the `kernels` version, PyTorch version, and platform build information.

```bash
export HF_HUB_DISABLE_TELEMETRY=1
```

