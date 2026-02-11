# kernels versions

Use `kernels versions` to list all available versions of a kernel on the Hub.

## Usage

```bash
kernels versions <repo_id>
```

## Examples

List versions of a kernel:

```bash
kernels versions kernels-community/activation
```

## Example Output

```text
Version 1: torch210-cu128-x86_64-windows, torch210-cxx11-cu126-x86_64-linux, torch210-cxx11-cu128-x86_64-linux, torch210-cxx11-cu130-x86_64-linux, torch210-metal-aarch64-darwin ✅, torch27-cxx11-cu118-x86_64-linux, torch27-cxx11-cu126-x86_64-linux, torch27-cxx11-cu128-aarch64-linux, torch27-cxx11-cu128-x86_64-linux, torch28-cxx11-cu126-aarch64-linux, torch28-cxx11-cu126-x86_64-linux, torch28-cxx11-cu128-aarch64-linux, torch28-cxx11-cu128-x86_64-linux, torch28-cxx11-cu129-aarch64-linux, torch28-cxx11-cu129-x86_64-linux, torch29-cxx11-cu126-aarch64-linux, torch29-cxx11-cu126-x86_64-linux, torch29-cxx11-cu128-aarch64-linux, torch29-cxx11-cu128-x86_64-linux, torch29-cxx11-cu130-aarch64-linux, torch29-cxx11-cu130-x86_64-linux, torch29-metal-aarch64-darwin
```

## Use Cases

- Check which versions are available before locking dependencies
- Find the latest version of a kernel
- Identify version SHAs for pinning in `pyproject.toml`

## See Also

- [kernels lock](cli-lock.md) - Lock kernel versions in your project
- [kernels download](cli-download.md) - Download locked kernels
