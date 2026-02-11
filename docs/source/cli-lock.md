# kernels lock

Use `kernels lock` to generate a `kernels.lock` file that pins kernel dependencies to specific revisions.

## Usage

```bash
kernels lock <project_dir>
```

## What It Does

- Reads kernel dependencies from `pyproject.toml` under `[tool.kernels.dependencies]`
- Resolves each kernel to its current revision SHA
- Writes a `kernels.lock` file with pinned versions and variant information

## Examples

Lock kernels in the current project:

```bash
kernels lock .
```

Lock kernels in a specific project:

```bash
kernels lock /path/to/my-project
```

## pyproject.toml Format

Add your kernel dependencies to `pyproject.toml`:

```toml
[tool.kernels.dependencies]
"kernels-community/activation" = 1
```

The version can be:

- A version number (e.g., `1`, `2`)

## kernels.lock Format

The generated lock file contains:

```json
[
  {
    "repo_id": "kernels-community/activation",
    "sha": "ece277f908b9453112722d584fee4b5696f21c49",
    "variants": {
      "torch210-cu128-x86_64-windows": {
        "hash": "sha256-cbf085e1d297d990d9cb074fb5079ff48e9682c729f53a0899a36b5164a6fb45",
        "hash_type": "git_lfs_concat"
      },
      // ...
      "torch29-metal-aarch64-darwin": {
        "hash": "sha256-9f665b54a53246a7d3627422f8a0d41d7956dc5409043dbd14c4ec0327aea310",
        "hash_type": "git_lfs_concat"
      }
    }
  }
]
```

## Workflow

1. Add dependencies to `pyproject.toml`
2. Run `kernels lock .` to generate the lock file
3. Commit both `pyproject.toml` and `kernels.lock`
4. Use `kernels download .` to install locked kernels

## See Also

- [kernels download](cli-download.md) - Download locked kernels
- [kernels versions](cli-versions.md) - View available kernel versions
