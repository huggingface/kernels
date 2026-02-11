# kernels generate-readme

Use `kernels generate-readme` to automatically generate documentation snippets for a kernel's public functions.

## Usage

```bash
kernels generate-readme <repo_id> [--revision <rev>]
```

## What It Does

- Downloads the specified kernel from the Hub
- Inspects the kernel's public API
- Generates markdown documentation snippets showing function signatures and usage

## Examples

Generate README snippets for a kernel:

```bash
kernels generate-readme kernels-community/activation > README.md
```

## Example Output

README.md snippet for `kernels-community/activation`:
```md
---
tags:
- kernels
---

## Functions

### Function `fatrelu_and_mul`

`(out: torch.Tensor, x: torch.Tensor, threshold: float = 0.0) -> None`

No documentation available.

### Function `gelu`

`(out: torch.Tensor, x: torch.Tensor) -> None`

No documentation available.

### Function `gelu_and_mul`

`(out: torch.Tensor, x: torch.Tensor) -> None`

No documentation available.

### Function `gelu_fast`

`(out: torch.Tensor, x: torch.Tensor) -> None`

No documentation available.

...
```