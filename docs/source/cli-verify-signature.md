# kernels verify-signature

Use `kernels verify-signature` to verify the metadata signature and check
that kernel files match the digest embedded in the metadata.

## Usage

```bash
kernels verify-signature <repo_id> <version> [--all-variants] \
  [--filter-unsigned] [--filter-no-digest]
```

## What It Does

- Checks that the signing identity in `metadata.json.sigstore` is approved.
- Verifies that `metadata.json` is not tampered with, using the signature
  in `metadata.json.sigstore`.
- Verifies that other kernel files are not tampered with, using the digest
  in `metadata.json`.

## Examples

Verify version `1` of the `kernels-community/relu` kernel. Only checks
the variant that is compatible with the current system:

```bash
kernels verify-signature kernels-community/relu 1
```

Verify all build variants of the same kernel:

```bash
kernels verify-signature kernels-community/relu 1 --all-variants
```

## Options

| Option               | Description                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--all-variants`     | Verify all build variants of each kernel instead of just the variant that is compatible with the current system. |
| `--filter-no-digest` | Skip variants that do not have a digest in the metadata (typically older builds that precede code signing).      |
| `--filter-unsigned`  | Skip variants that do not have a detached signature (typically older builds that precede code signing).          |
