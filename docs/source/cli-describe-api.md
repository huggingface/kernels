# kernels describe-api

Use `kernels describe-api` to list a kernel's public functions and layers from
the `symbols.json` generated at build time.

For Hub kernels, the resolver selects the matching build variant and downloads
only `metadata.json` and `symbols.json`. Kernel source files and binaries are
not downloaded or imported.

Online Hub inspection requires a trusted kernel publisher by default. Pass
`--trust-remote-code` to explicitly allow a kernel from another publisher.

## Usage

```bash
kernels describe-api <repo_id_or_path>
```

The local path can point to a kernel repository containing `build/`, the
`build/` directory itself, or an individual build variant containing
`metadata.json` and `symbols.json`.

Options:

- `--revision <revision>`: inspect a specific branch, tag, or commit (Hub only).
- `--version <version>`: inspect a specific integer kernel version (Hub only).
  Cannot be combined with `--revision`. By default, the latest numbered version
  is used. Repositories without numbered versions require an explicit `--revision`.
- `--json`: print the full symbol data, including signatures, docstrings, layer
  attributes, and the selected source and variant, as JSON.
- `--trust-remote-code`: allow kernels from untrusted publishers (Hub only).
  Disabled by default.


## Examples

```bash
kernels describe-api kernels-community/liger-kernels
kernels describe-api kernels-community/liger-kernels --version 2
kernels describe-api kernels-community/liger-kernels --revision main --json
kernels describe-api sayakpaul/liger-kernels --trust-remote-code
kernels describe-api ./liger-kernels
kernels describe-api ./liger-kernels/build/torch-cuda
```

## Example output

```text
Repository: kernels-community/liger-kernels
Revision: v2
Variant: torch-cuda

Functions:
  LigerForCausalLMLoss
  liger_rotary_pos_emb

Layers:
  LigerRMSNorm  has_backward=unknown  can_torch_compile=unknown
  LigerLinear  has_backward=unknown  can_torch_compile=True
  LigerSwiGLUMLP  has_backward=unknown  can_torch_compile=True
  LigerGEGLUMLP  has_backward=unknown  can_torch_compile=True
  LigerTiledSwiGLUMLP  has_backward=unknown  can_torch_compile=True
  LigerTiledGEGLUMLP  has_backward=unknown  can_torch_compile=True
```

Functions are the public exports in the kernel's top-level `__all__`; layers
are the public classes in `layers.__all__`. The generator excludes private
and testing-only symbols. Layer attributes recorded as `null` are displayed
as `unknown`. Empty groups are displayed as `(none)`.

For kernel metadata, use [kernels info](cli-info.md). For available build
variants, use [kernels variants](cli-variants.md).
