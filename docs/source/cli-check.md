# kernels check

Use `kernels check` to verify that a kernel on the Hub meets compliance requirements.

## What It Checks

- Python ABI compatibility (default: 3.9)
- Operating system compatibility (macOS 15.0+, manylinux_2_28)

## Usage

```bash
kernels check <repo_id> [--revision <rev>] [--macos <version>] [--manylinux <version>] [--python-abi <version>]
```

## Installation

`kernels check` requires an additional dependency:

```bash
uv pip install kernel-abi-check # or pip install kernel-abi-check
```

## Examples

Check a kernel on the Hub:

```bash
kernels check kernels-community/flash-attn3
```

Check a specific revision:

```bash
kernels check kernels-community/flash-attn3 --revision v2
```

Check with custom compatibility requirements:

```bash
kernels check kernels-community/flash-attn3 --python-abi 3.10 --manylinux manylinux_2_31
```

## Example Output

```text
Checking variant: torch210-metal-aarch64-darwin
  Dynamic library _example_kernel_metal_2juixjwdznbhy.abi3.so:
    🐍 Python ABI 3.9 compatible
    🍏 compatible with macOS 15.0
Checking variant: torch29-metal-aarch64-darwin
  Dynamic library _example_kernel_metal_vtlnpevkb6uum.abi3.so:
    🐍 Python ABI 3.9 compatible
    🍏 compatible with macOS 15.0
```

## Options

| Option         | Default          | Description                         |
| -------------- | ---------------- | ----------------------------------- |
| `--revision`   | `main`           | Branch, tag, or commit SHA to check |
| `--macos`      | `15.0`           | Minimum macOS version to require    |
| `--manylinux`  | `manylinux_2_28` | Manylinux version to require        |
| `--python-abi` | `3.9`            | Python ABI version to require       |

