# E2E Test Plan: kernel-builder init + build + upload + get_kernel

## Goal

Validate the full lifecycle: `kernel-builder init` creates a valid project, the template builds successfully, the built kernel can be uploaded to the Hub, and `get_kernel()` can download and use it.

---

## Overview

A single GitHub Actions workflow with three jobs for fault isolation:

1. **init-and-build** (Nix runner) -- `kernel-builder init`, validate scaffold, `nix build` one CUDA variant.
2. **upload** (GPU runner) -- `kernel-builder upload` the built artifacts to Hub.
3. **download-and-test** (GPU runner) -- `get_kernel()` download + correctness check, cleanup.

If init+build fails, we know the template or build infra is broken. If upload fails, the Rust upload logic is broken. If download+test fails, the Python `get_kernel()` path is broken. Each job's failure points to a specific component.

---

## Speed considerations

- **Build exactly one variant.** Use the `ci` Nix target which selects one variant per framework (one CUDA variant in our case).
- **No matrix.** Single Python version (3.12), single Torch version (latest: 2.10), single backend (CUDA).
- **Cachix.** Leverage the existing `huggingface` Cachix cache so Nix derivations are fetched, not rebuilt.
- **Compile `kernel-builder` with `--release` once** in the build job and pass the binary via artifact to the GPU jobs (avoids compiling Rust twice).

---

## Trigger paths

The workflow should run on changes to the core components that affect this lifecycle:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    paths:
      - "kernel-builder/**"      # CLI: init, build, upload commands
      - "kernels/src/**"         # Python library: get_kernel, variants, install
      - "nix-builder/**"         # Nix build infrastructure
      - "kernels-data/**"        # Shared config/data structures
  workflow_dispatch:
```

This keeps the e2e tests focused on changes that could actually break the init/build/upload/download cycle, avoiding unnecessary runs on docs-only or CI config changes.

---

## Job 1: `init-and-build`

### What it validates

- `kernel-builder init` produces a complete, well-formed project.
- The generated `build.toml` is valid.
- The project builds via Nix without errors.
- The build produces a working kernel variant.

### Implementation

**Runner:** `aws-highmemory-32-plus-nix` (matches existing `build_kernel.yaml`)

#### Step 1: Build `kernel-builder` CLI

```bash
cargo build --release --manifest-path kernel-builder/Cargo.toml
```

#### Step 2: Init a fresh kernel project

```bash
cd /tmp
kernel-builder init --name kernels-test/e2e-test-kernel --backends cuda
```

Creates `/tmp/e2e-test-kernel/` with the full scaffold.

#### Step 3: Validate the scaffold

Quick checks that key files exist:

```bash
cd /tmp/e2e-test-kernel
test -f build.toml
test -f flake.nix
test -f torch-ext/e2e_test_kernel/__init__.py
test -f torch-ext/torch_binding.cpp
test -f torch-ext/torch_binding.h
test -f e2e_test_kernel_cuda/e2e_test_kernel.cu
test -f tests/test_e2e_test_kernel.py
test -f example.py
```

Verify key fields in `build.toml`:
- `name = "e2e-test-kernel"`
- `repo-id = "kernels-test/e2e-test-kernel"`
- `backend = "cuda"` appears in a kernel section

#### Step 4: Patch flake.nix for local nix-builder

The init template generates a `flake.nix` that references `github:huggingface/kernels` (the remote repo). For CI, we need it to use the **local checkout** so the test validates the current code:

```bash
cd /tmp/e2e-test-kernel
sed -i 's|github:huggingface/kernels|path:'"$GITHUB_WORKSPACE"'|' flake.nix
```

This changes `kernel-builder.url = "github:huggingface/kernels"` to `kernel-builder.url = "path:/path/to/checkout"`, matching how the example kernels work (`path:../../..`).

#### Step 5: Build one CUDA variant with Nix

```bash
cd /tmp/e2e-test-kernel
nix build .#ci -L
cp -rL result/* build/
```

The `ci` Nix target builds exactly one variant per framework. This is the fastest path -- it's the same target the existing `build_kernel.yaml` CI uses.

#### Step 6: Verify build artifacts

```bash
VARIANT_DIR=$(ls -d build/torch* | head -1)
test -f "$VARIANT_DIR/__init__.py"
test -f "$VARIANT_DIR/metadata.json"
ls "$VARIANT_DIR"/*.so  # At least one shared object
```

#### Step 7: Upload artifacts

Upload the built kernel directory and `kernel-builder` binary for downstream jobs.

---

## Job 2: `upload`

### What it validates

- `kernel-builder upload` successfully pushes build artifacts to a Hub repo.

### Implementation

**Runner:** `aws-g6-12xlarge-plus` (GPU runner -- needed for Job 3 anyway, reuse the same runner group)
**Depends on:** `init-and-build`

#### Step 1: Download artifacts from Job 1

Download the built kernel directory and `kernel-builder` binary.

#### Step 2: Upload kernel to Hub

```bash
kernel-builder upload /tmp/e2e-test-kernel \
  --repo-id kernels-test/kernels-upload-test \
  --private
```

Uses `HF_TOKEN` secret for authentication.

---

## Job 3: `download-and-test`

### What it validates

- `get_kernel()` can download the uploaded kernel and resolve the correct variant.
- The imported module is callable and produces correct results.

### Implementation

**Runner:** `aws-g6-12xlarge-plus` (GPU runner)
**Depends on:** `upload`

#### Step 1: Install Python deps

```bash
cd kernels
uv sync --all-extras --dev
uv pip install torch==2.10.0
```

#### Step 2: Download and test via `get_kernel()`

```python
import torch
from kernels import get_kernel

kernel = get_kernel("kernels-test/kernels-upload-test", version=1)

x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
result = kernel.e2e_test_kernel(x)
expected = x + 1.0
torch.testing.assert_close(result, expected)
print("E2E test passed!")
```

#### Step 3: Cleanup (always runs)

Delete the Hub repo regardless of test outcome:

```python
from huggingface_hub import HfApi
api = HfApi()
api.delete_repo("kernels-test/kernels-upload-test")
```

---

## Key risks and mitigations

| Risk | Mitigation |
|------|------------|
| Init template `flake.nix` points to remote nix-builder | `sed` patch to use `path:$GITHUB_WORKSPACE` |
| Leftover repos on failure | `if: always()` cleanup step in Job 3 |
| Nix build is slow | Use `ci` target (one variant) + Cachix caching |
| Rust recompilation on GPU runner | Pass pre-built binary via artifact |

---

## Complete workflow file

```yaml
name: "E2E: kernel-builder init + build + upload + download"

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
    paths:
      - "kernel-builder/**"
      - "kernels/src/**"
      - "nix-builder/**"
      - "kernels-data/**"
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  init-and-build:
    name: Init and build kernel
    runs-on:
      group: aws-highmemory-32-plus-nix
    steps:
      - uses: actions/checkout@v6
      - uses: DeterminateSystems/nix-installer-action@main
        with:
          extra-conf: |
            max-jobs = 8
            cores = 12
            sandbox-fallback = false
      - uses: cachix/cachix-action@v16
        with:
          name: huggingface
          authToken: "${{ secrets.CACHIX_AUTH_TOKEN }}"
        env:
          USER: runner

      - name: Build kernel-builder CLI
        run: cargo build --release --manifest-path kernel-builder/Cargo.toml

      - name: Init kernel project
        run: |
          cd /tmp
          $GITHUB_WORKSPACE/target/release/kernel-builder init \
            --name kernels-test/e2e-test-kernel \
            --backends cuda

      - name: Validate scaffold
        run: |
          cd /tmp/e2e-test-kernel
          test -f build.toml
          test -f flake.nix
          test -f torch-ext/e2e_test_kernel/__init__.py
          test -f torch-ext/torch_binding.cpp
          test -f torch-ext/torch_binding.h
          test -f e2e_test_kernel_cuda/e2e_test_kernel.cu
          test -f tests/test_e2e_test_kernel.py
          test -f example.py
          grep -q 'name = "e2e-test-kernel"' build.toml
          grep -q 'repo-id = "kernels-test/e2e-test-kernel"' build.toml
          grep -q 'backend = "cuda"' build.toml

      - name: Patch flake.nix to use local nix-builder
        run: |
          cd /tmp/e2e-test-kernel
          sed -i 's|github:huggingface/kernels|path:'"$GITHUB_WORKSPACE"'|' flake.nix

      - name: Build one CUDA variant
        run: |
          cd /tmp/e2e-test-kernel
          nix build .#ci -L
          mkdir -p build
          cp -rL result/* build/

      - name: Verify build artifacts
        run: |
          cd /tmp/e2e-test-kernel
          VARIANT_DIR=$(ls -d build/torch* | head -1)
          echo "Built variant: $VARIANT_DIR"
          test -f "$VARIANT_DIR/__init__.py"
          test -f "$VARIANT_DIR/metadata.json"
          ls "$VARIANT_DIR"/*.so

      - name: Upload built kernel
        uses: actions/upload-artifact@v6
        with:
          name: e2e-built-kernel
          path: /tmp/e2e-test-kernel/

      - name: Upload kernel-builder binary
        uses: actions/upload-artifact@v6
        with:
          name: kernel-builder-bin
          path: ${{ github.workspace }}/target/release/kernel-builder

  upload:
    name: Upload kernel to Hub
    needs: init-and-build
    runs-on:
      group: aws-g6-12xlarge-plus
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
    steps:
      - name: Download built kernel
        uses: actions/download-artifact@v7
        with:
          name: e2e-built-kernel
          path: /tmp/e2e-test-kernel

      - name: Download kernel-builder binary
        uses: actions/download-artifact@v7
        with:
          name: kernel-builder-bin
          path: /tmp/bin

      - name: Make binary executable
        run: chmod +x /tmp/bin/kernel-builder

      - name: Upload kernel to Hub
        run: |
          /tmp/bin/kernel-builder upload /tmp/e2e-test-kernel \
            --repo-id kernels-test/kernels-upload-test \

  download-and-test:
    name: Download and test kernel via get_kernel
    needs: upload
    runs-on:
      group: aws-g6-12xlarge-plus
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      UV_PYTHON_PREFERENCE: only-managed
    steps:
      - uses: actions/checkout@v6

      - name: Install uv and set Python version
        uses: astral-sh/setup-uv@v7
        with:
          python-version: "3.12"

      - name: Install Python deps
        working-directory: ./kernels
        run: |
          uv sync --all-extras --dev
          uv pip install torch==2.10.0

      - name: Test get_kernel download and usage
        working-directory: ./kernels
        run: |
          uv run python -c "
          import torch
          from kernels import get_kernel

          kernel = get_kernel('kernels-test/kernels-upload-test', version=1)

          x = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
          result = kernel.e2e_test_kernel(x)
          expected = x + 1.0
          torch.testing.assert_close(result, expected)
          print('E2E test passed: get_kernel + correctness check')
          "

      - name: Cleanup Hub repo
        if: always()
        working-directory: ./kernels
        run: |
          uv run python -c "
          from huggingface_hub import HfApi
          api = HfApi()
          try:
              api.delete_repo('kernels-test/kernels-upload-test')
              print('Cleaned up repo')
          except Exception as e:
              print(f'Cleanup warning: {e}')
          "
```

---

## Files to create

| File | Purpose |
|------|---------|
| `.github/workflows/test_e2e.yaml` | The workflow above |

No new Python test files needed -- the e2e test is self-contained in the workflow.

---

## Execution flow

```
Job 1: init-and-build (Nix runner, no GPU)
  1. cargo build kernel-builder
  2. kernel-builder init --backends cuda
  3. validate scaffold files + build.toml
  4. patch flake.nix → local nix-builder
  5. nix build .#ci (one CUDA variant)
  6. verify artifacts (*.so, metadata.json)
  → artifacts: built kernel dir + kernel-builder binary

Job 2: upload (GPU runner)
  7. kernel-builder upload → Hub
  ✗ Failure here = upload logic is broken

Job 3: download-and-test (GPU runner)
  8. get_kernel() → download + import
  9. call kernel function + assert correctness
  10. delete Hub repo (always)
  ✗ Failure here = Python get_kernel / variant resolution is broken
```
