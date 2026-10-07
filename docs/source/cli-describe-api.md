# kernels describe-api

Use `kernels describe-api` to list a kernel's public functions and layers without
loading the kernel or installing its dependencies. For Hub kernels, the command
downloads only the Python source needed to inspect the API, plus `metadata.json`
when needed to locate an older build's package. It does not download compiled
libraries or a repository snapshot.

## Usage

```bash
kernels describe-api <repo_id_or_path>
```

Options:

- `--version <version>`: select an integer kernel version or `latest` (the default).
- `--revision <revision>`: select a branch, tag, or commit instead of a version.
- `--json`: print the description as JSON for machine-readable output.

`--version` and `--revision` are mutually exclusive and apply only to Hub repositories. The
latest numbered version is selected by default, or `main` for an unversioned
repository.

An existing local directory is treated as a kernel.

## Examples

```bash
kernels describe-api kernels-community/activation
kernels describe-api kernels-community/activation --version 1
kernels describe-api kernels-community/activation --revision main
kernels describe-api ./my-kernel
kernels describe-api kernels-community/activation --json
```

## Export conventions

Functions must be listed in the top-level module's `__all__`. Layers must be
listed in `layers.py` or `layers/__init__.py`'s own `__all__`. A module without
`__all__` contributes no exports. Private names and symbols from
`_private_for_testing` are purposefully excluded.

Each layer includes its explicitly declared `has_backward` and
`can_torch_compile` boolean attributes.

For example, a kernel declaring `gelu` and `Gelu` in the respective `__all__`
lists could produce:

```text
Repository: example/activation
Revision: v1

Functions:
  gelu

Layers:
  Gelu  has_backward=True  can_torch_compile=True
```

With `--json`, the same description is:

```json
{
  "repo_id": "example/activation",
  "revision": "v1",
  "functions": ["gelu"],
  "layers": [
    {"name": "Gelu", "has_backward": true, "can_torch_compile": true}
  ]
}
```

Local kernels include `path` instead of `repo_id` and `revision`.

Use [kernels info](cli-info.md) for metadata such as the license, dependencies,
and supported backends, and [kernels variants](cli-variants.md) for build variants
and compatibility information.
