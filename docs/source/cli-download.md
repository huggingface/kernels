# kernels download

Use `kernels download` to download kernels that have been locked in a project's `kernels.lock` file.

## Usage

```bash
kernels download <project_dir> [--all-variants]
```

## What It Does

- Reads the `kernels.lock` file from the specified project directory
- Downloads each locked kernel at its pinned revision (SHA)
- Installs the appropriate variant for your platform (or all variants with `--all-variants`)

## Examples

Download kernels for the current project:

```bash
kernels download .
```

Download all build variants (useful for CI or multi-platform builds):

```bash
kernels download . --all-variants
```

Download kernels for a specific project:

```bash
kernels download /path/to/my-project
```

## Options

| Option           | Description                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `--all-variants` | Download all build variants of each kernel instead of just the current platform's variant |

## Prerequisites

Your project directory must contain a `kernels.lock` file. Generate one using [`kernels lock`](cli-lock.md).

## See Also

- [kernels lock](cli-lock.md) - Generate the lock file
- [kernels versions](cli-versions.md) - View available kernel versions
