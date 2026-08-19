# Ship Triton autotune configurations

Triton kernels typically have parameters, such as tile sizes, number of warps and 
number of pipeline stages, whose optimal values depend on the GPU and the
problem shape. Autotuning finds good values for these parameters, but doing
it at runtime (e.g. with the
[`@triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
decorator) re-benchmarks every candidate configuration in each new process.


However, one can run the tuner for the GPU models they like and store the
best found configurations as files for using them later. This effectively
reduces the potentially costly tuning time.

`kernel-builder` support this by packaging these configurations as JSON files
with the kernel. At runtime, the kernel looks up the configuration for the
current GPU and shape and falls back to sensible defaults when there is no
matching configuration.

This is how, for example, the vLLM fused MoE kernel ships on the Hub: the
[`RedHatAI/moe`](https://huggingface.co/RedHatAI/moe/tree/main/torch-ext/moe/configs)
repository contains a `configs` directory with tuned configurations for many
GPUs. This page walks through a small, complete example of the same pattern:
the [`gemm-triton-autotune`](https://github.com/huggingface/kernels/tree/main/examples/kernels/gemm-triton-autotune)
example kernel, a Triton GEMM published as
[`kernels-test/gemm-triton-autotune`](https://huggingface.co/kernels-test/gemm-triton-autotune).

## Shipping data files with a kernel

Since autotune files are plain JSON files, we can store them anywhere inside the
kernels main Python sources in `torch-ext/<kernel_name>`. For this example,
we will use  `torch-ext/<kernel_name>/configs/`. By default, only `py` and `pyi`
files are picked up from the kenel's Python source directory, so add `json` to the
[`pyext` option](writing-kernels.md#torch-noarch) in `build.toml`:

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
pyext = ["json", "py"]
```

## Configuration file layout

A GEMM computes `(M, K) @ (K, N)`. For a model, the weight dimensions `N`
and `K` are known ahead of time, while `M` (e.g. the number of tokens)
varies at runtime. The example therefore stores one file per `(N, K)` shape
and GPU, following the same naming convention as the MoE kernel:

```
configs/N=4096,K=4096,device_name=NVIDIA_L4.json
configs/N=14336,K=4096,device_name=NVIDIA_L4.json
```

Each file maps an `M` value to the best configuration that the autotuner
found for that `M`:

```json
{
    "1": {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3
    },
    "1024": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3
    }
}
```

Since the device name is part of the file name, configurations tuned for one
GPU are never applied to another. A configuration that would exceed the
resources of a smaller GPU (e.g. shared memory) is therefore harmless to
ship.

## Looking up configurations at runtime

At kernel launch, the kernel checks whether a configuration file exists for
the current device and shape. If it does, the configuration with the nearest
tuned `M` is used; otherwise the kernel falls back to a conservative default
and logs a warning. The lookup is cached, so the file is read at most once
per process:

```python
@functools.cache
def _load_tuned_configs(N: int, K: int) -> Optional[Dict[int, Dict[str, int]]]:
    path = _CONFIGS_DIR / config_file_name(N, K)
    if path.exists():
        logger.info("Using tuned GEMM configurations from %s.", path)
        with open(path) as f:
            return {int(m): config for m, config in json.load(f).items()}
    logger.warning(
        "No tuned GEMM configuration found for this device and shape (%s). "
        "Falling back to heuristic defaults, performance may be suboptimal. "
        "Generate a configuration with `tune_gemm(N=%d, K=%d)`.",
        path.name,
        N,
        K,
    )
    return None


def get_config(M: int, N: int, K: int) -> Dict[str, int]:
    tuned = _load_tuned_configs(N, K)
    if tuned is not None:
        # Tuned Ms are spaced logarithmically, so pick the nearest in log space.
        nearest_m = min(tuned, key=lambda m: abs(math.log(M / m)))
        return tuned[nearest_m]
    return default_config(M, N, K)
```

The configuration is then passed to the Triton kernel as its `constexpr`
and launch parameters (see
[`gemm.py`](https://github.com/huggingface/kernels/blob/main/examples/kernels/gemm-triton-autotune/torch-ext/gemm_triton_autotune/gemm.py)
in the example):

```python
def launch_gemm_kernel(a, b, out, config):
    M, K = a.shape
    N = b.shape[1]
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_kernel[grid](a, b, out, M, N, K, ..., **config)
```

## Generating the configurations

The autotuner itself is ordinary benchmarking code: for each `M`, benchmark
every candidate configuration with `triton.testing.do_bench` and keep the
fastest. The example ships the tuner as part of the kernel (the `tune_gemm`
function in
[`tuning.py`](https://github.com/huggingface/kernels/blob/main/examples/kernels/gemm-triton-autotune/torch-ext/gemm_triton_autotune/tuning.py)),
so users can also generate configurations for GPUs that the kernel author
did not tune. Candidates that do not fit the device (Triton raises
`OutOfResources`) are skipped.

The example repository contains a small script,
[`tune.py`](https://github.com/huggingface/kernels/blob/main/examples/kernels/gemm-triton-autotune/tune.py),
that runs the tuner and writes the configuration files to the source tree:

```bash
$ python tune.py --n 4096 --k 4096
$ python tune.py --n 14336 --k 4096
```

Commit the generated files, rebuild, and the configurations ship with the
kernel. When tuning a kernel that is not yet published, build it locally
(see [Develop locally](local-dev.md)) and point `LOCAL_KERNELS` at the
build:

```bash
$ LOCAL_KERNELS=kernels-test/gemm-triton-autotune=build python tune.py --n 4096 --k 4096
```

## Impact

Tuned configurations are cheap to ship and can make a large difference. On
an NVIDIA L4, the tuned configuration for a `(1024, 4096) @ (4096, 4096)`
float16 GEMM is ~1.4× faster than the example's heuristic default (0.50 ms
vs. 0.71 ms) — on par with cuBLAS for this shape.
