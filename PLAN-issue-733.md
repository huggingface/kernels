# Plan: Triton autotune configs example (issue #733)

**Issue:** [huggingface/kernels#733](https://github.com/huggingface/kernels/issues/733) —
"Document/example: shipping Triton autotune configs with a kernel"

**Goal:** Add an example Triton kernel to `examples/kernels/` that ships pre-computed
autotune configurations as JSON files, plus a script that generates those files and
documentation explaining the pattern. Reference implementation:
[`RedHatAI/moe`](https://huggingface.co/RedHatAI/moe/tree/main/torch-ext/moe) (vLLM fused MoE).

## Background / findings from the repo

- `examples/kernels/relu-triton/` is the existing Triton example: a `torch-noarch`
  kernel with `torch.library.custom_op` wrappers. It is built in CI for CUDA, ROCm,
  and XPU via the `ciKernels` / `ciRocmKernels` / `ciXpuKernels` lists in
  `examples/kernels/flake.nix`, and its tests run on GPU through
  `nix-builder/tests/Dockerfile.test-kernel` + `run-tests.sh` using the
  `LOCAL_KERNELS` env var (`kernels-test/<repo>=<built path>`).
- `examples/kernels/extra-data/` already demonstrates shipping non-Python data by
  adding `"json"` to `pyext` in `build.toml` — the exact mechanism we need for
  config files.
- The vLLM MoE pattern (`fused_moe.py`): config files live in
  `torch-ext/moe/configs/` named
  `E=...,N=...,device_name=NVIDIA_H100_80GB_HBM3[,dtype=...].json`. Each file maps a
  batch-size bucket (`M`) to a Triton config dict (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`,
  `num_warps`, `num_stages`). At runtime an `lru_cache`d loader looks up the file for
  the current device; if absent, it falls back to a heuristic default and logs a
  warning.
- A local NVIDIA L4 GPU is available — the same GPU as the CI test runners
  (`aws-g6-12xlarge`) — so we can generate and commit a real config that CI will
  actually exercise.

## Design decisions

1. **Kernel: a Triton GEMM (`matmul`)**, not another relu. Autotuning is only
   meaningful when tile sizes / warps / stages matter; GEMM is the canonical case and
   mirrors the MoE reference. Kernel body can be the standard Triton matmul tutorial
   kernel (fp16/bf16/fp32 inputs, fp16-accumulate-in-fp32).
2. **Ship JSON lookup tables instead of relying on `@triton.autotune`.** The
   decorator re-benchmarks in every process (slow startup, nondeterministic) and its
   cache is not portable. Instead: an offline tuning script does the
   `@triton.autotune`-style grid search once, writes JSON, and the kernel loads the
   JSON at call time. This is exactly what the issue asks to demonstrate.
3. **File naming convention** (adapted from vLLM MoE): since GEMM weight dims are
   known ahead of time but the batch dim `M` varies at runtime, one file per
   `(N, K, device)`: `configs/N=4096,K=4096,device_name=NVIDIA_L4.json`, whose keys
   are `M` values and values are config dicts. Runtime picks the nearest `M` key
   (same `min(keys, key=|log(M/key)|)` trick as vLLM).
4. **Tuning entry point ships with the kernel** (a `tune.py` module inside the
   package) so users can regenerate configs for *their* GPU after
   `get_kernel(...)`, plus a thin CLI script in the example root for kernel authors.

## New files

```
examples/kernels/gemm-triton-autotune/
├── build.toml                      # torch-noarch, pyext = ["py", "json"]
├── flake.nix                       # copied from relu-triton
├── CARD.md                         # copied from relu-triton
├── tune.py                         # CLI: python tune.py --n 4096 --k 4096 [--out ...]
│                                   # writes into torch-ext/gemm_triton_autotune/configs/
├── torch-ext/gemm_triton_autotune/
│   ├── __init__.py                 # exports gemm(), tune_gemm()
│   ├── gemm.py                     # @triton.jit matmul kernel + custom_op wrapper;
│   │                               # per-call config lookup via tuning.get_config()
│   ├── tuning.py                   # device-name helper, config file naming,
│   │                               # lru_cache'd JSON loader, default-config fallback
│   │                               # (with one-time warning), tune_gemm() grid search
│   │                               # using triton.testing.do_bench
│   └── configs/
│       └── N=4096,K=4096,device_name=NVIDIA_L4.json   # generated on local L4
└── tests/
    ├── __init__.py
    ├── conftest.py                 # device fixture, same as relu-triton
    └── test_gemm.py                # correctness vs torch.matmul across dtypes/shapes;
                                    # shipped-config load test (monkeypatched device name);
                                    # fallback-to-default test for unknown (N, K)
```

### `build.toml` sketch

```toml
[general]
name = "gemm-triton-autotune"
version = 1
edition = 5
license = "Apache-2.0"
backends = ["cuda", "rocm", "xpu"]

[general.hub]
repo-id = "kernels-test/gemm-triton-autotune"

[torch-noarch]
pyext = ["py", "json"]
```

### Runtime config lookup (in `tuning.py`)

```python
@functools.lru_cache
def get_config(M: int, N: int, K: int) -> dict:
    path = Path(__file__).parent / "configs" / _config_file_name(N, K)
    if path.exists():
        configs = {int(m): cfg for m, cfg in json.loads(path.read_text()).items()}
        return configs[min(configs, key=lambda m: abs(math.log(M / m)))]
    warnings.warn(f"No tuned GEMM config for {path.name}, using defaults...", once)
    return _default_config(M, N, K)
```

### Tuning script behavior

- Candidate grid: the usual matmul space (`BLOCK_M/N/K ∈ {32..256}`, `GROUP_M`,
  `num_warps ∈ {4, 8}`, `num_stages ∈ {2..5}`), pruned to valid combos.
- Benchmarks each candidate with `triton.testing.do_bench` for each `M` in
  `{1, 16, 64, 256, 1024, 4096}` at fixed `(N, K)`.
- Writes `{M: best_config}` JSON to the package `configs/` dir, named with
  `torch.cuda.get_device_name().replace(" ", "_")` (XPU equivalent when applicable).

## Existing files to modify

1. **`examples/kernels/flake.nix`** — register the new kernel in `ciKernels`
   (`torch-cuda` noarch build, like `relu-triton-kernel`), `ciRocmKernels`, and
   `ciXpuKernels`.
2. **`.github/workflows/build_kernel.yaml`** — add `gemm-triton-autotune-kernel` to
   the uploaded-artifacts list.
3. **`nix-builder/tests/Dockerfile.test-kernel`** — `COPY
   examples/kernels/gemm-triton-autotune/tests ./gemm_triton_autotune_tests`.
4. **`nix-builder/tests/run-tests.sh`** — run the new tests with
   `LOCAL_KERNELS="kernels-test/gemm-triton-autotune=..."`.
5. **Docs:**
   - New page `docs/source/builder/triton-autotune.md` — "Shipping Triton autotune
     configurations": why ship configs, the JSON-per-device pattern, `pyext = ["json"]`,
     the loader/fallback pattern, how to run the tune script, links to the example
     and to `RedHatAI/moe`.
   - Add the page to `docs/source/_toctree.yml` (kernel-builder section, after
     `builder/writing-kernels`).

## Implementation order

1. Scaffold the example kernel (build.toml, flake.nix, CARD.md, package code).
2. Set up a local venv (`uv venv` + torch/cu126 + triton + kernels + pytest) and get the
   kernel running directly from `torch-ext/` on the L4.
3. Run `tune.py` on the L4 for `(N=4096, K=4096)`; commit the generated JSON.
4. Write tests; run them locally against the local build
   (`LOCAL_KERNELS=kernels-test/gemm-triton-autotune=<path>` after a
   `kernels build`/nix build, matching the CI invocation).
5. Wire up CI (flake.nix lists, workflow artifact list, Dockerfile, run-tests.sh).
6. Write the docs page + toctree entry.
7. `nix flake check` / build the example via
   `nix build ./examples/kernels#ci-build-cuda` if feasible locally, else rely on CI.

## Out of scope / maintainer follow-ups

- Pushing the built kernel to the `kernels-test/gemm-triton-autotune` Hub repo
  (needed for `get_kernel` without `LOCAL_KERNELS`) requires org access — CI tests
  use `LOCAL_KERNELS`, so nothing blocks on this, but the Hub repo should be created
  when merging (same as other `kernels-test/*` examples).
- Configs for ROCm/XPU devices can be contributed later by whoever has the hardware;
  the fallback path covers them meanwhile (and the fallback is itself part of what
  the example demonstrates).

## Open questions

1. Kernel/package name: `gemm-triton-autotune` (proposed) vs `matmul-triton-tune`.
2. Should the docs page live under kernel-builder docs (proposed) or as a section
   appended to `writing-kernels.md`?
3. Single `(N, K)` shape for the committed config (proposed: 4096×4096) or a couple
   of shapes to show multiple config files?
