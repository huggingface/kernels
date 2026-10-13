# kernels describe-api

Use `kernels describe-api` to list a kernel's public functions and layers from
the `symbols.json` generated at build time. It reads the symbol file without
importing the kernel or downloading its code, so a compatible GPU and the
kernel's dependencies are not required.

## Usage

```bash
kernels describe-api <repo_id_or_path>
```

The local path can point to a kernel repository containing `build/`, the
`build/` directory itself, or an individual build variant containing
`symbols.json`.

Options:

- `--revision <revision>`: inspect a specific branch, tag, or commit (Hub only).
- `--version <version>`: inspect a specific integer kernel version (Hub only).
  Cannot be combined with `--revision`. By default, the latest numbered version
  is used, or `main` if the repository has no numbered versions.
- `--json`: print the full symbol data, including signatures, docstrings, layer
  attributes, and the selected source and variant, as JSON.

The command selects the first build variant in alphabetical order. If its
`symbols.json` is missing, it exits with an error; it does not try another
variant or version.

## Examples

```bash
kernels describe-api kernels-community/liger-kernels
kernels describe-api kernels-community/liger-kernels --version 2
kernels describe-api kernels-community/liger-kernels --revision main --json
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
