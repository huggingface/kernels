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

## Example Output

```bash
$ kernels verify-signature kernels-community/relu 1 --all-variants
❌ torch210-cpu-aarch64-darwin: cannot verify kernel integrity, signature not found
❌ torch210-cu128-x86_64-windows: cannot verify kernel integrity, signature not found
❌ torch210-cxx11-cpu-aarch64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cpu-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu126-aarch64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu126-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu128-aarch64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu128-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu130-aarch64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-cu130-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-rocm70-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-rocm71-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-cxx11-xpu20253-x86_64-linux: cannot verify kernel integrity, metadata does not have a digest
❌ torch210-metal-aarch64-darwin: cannot verify kernel integrity, signature not found
❌ torch210-xpu20253-x86_64-windows: cannot verify kernel integrity, signature not found
✅ torch211-cpu-aarch64-darwin: kernel metadata is correctly signed
❌ torch211-cu128-x86_64-windows: cannot verify kernel integrity, signature not found
✅ torch211-cxx11-cpu-aarch64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cpu-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu126-aarch64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu126-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu128-aarch64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu128-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu130-aarch64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-cu130-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-rocm71-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-rocm72-x86_64-linux: kernel metadata is correctly signed
✅ torch211-cxx11-xpu20253-x86_64-linux: kernel metadata is correctly signed
✅ torch211-metal-aarch64-darwin: kernel metadata is correctly signed
```

## Options

| Option               | Description                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--all-variants`     | Verify all build variants of each kernel instead of just the variant that is compatible with the current system. |
| `--filter-no-digest` | Skip variants that do not have a digest in the metadata (typically older builds that precede code signing).      |
| `--filter-unsigned`  | Skip variants that do not have a detached signature (typically older builds that precede code signing).          |
