"""Loading and generating tuned GEMM configurations.

Tuned configurations are stored as JSON files in the ``configs`` directory
of this package, one file per ``(N, K, device)`` combination. Each file maps
an ``M`` value (the dimension that is typically only known at runtime, e.g.
the number of tokens) to the Triton configuration that performed best for
that ``M``:

    {
        "16": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, ..., "num_warps": 4},
        "1024": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, ...},
        ...
    }

At kernel launch, the configuration with the nearest tuned ``M`` is used.
When no configuration file exists for the current device and shape, a
heuristic default is used instead and a warning is logged once.
"""

import functools
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import triton

logger = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).parent / "configs"

# The M values that are tuned by default: GEMMs are often skinny (a few
# tokens during decoding) or wide (large batches during prefill), so we
# cover a logarithmic range.
DEFAULT_TUNE_MS = (1, 16, 64, 256, 1024, 4096)


def device_name() -> str:
    """Name of the current accelerator, as used in configuration file names."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        name = torch.xpu.get_device_name()
    else:
        name = torch.cuda.get_device_name()
    return name.replace(" ", "_")


def config_file_name(N: int, K: int) -> str:
    return f"N={N},K={K},device_name={device_name()}.json"


@functools.lru_cache
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
    """Return the best known configuration for a ``(M, N, K)`` GEMM."""
    tuned = _load_tuned_configs(N, K)
    if tuned:
        # Pick the configuration tuned for the M closest to the runtime M
        # (in log space, since tuned Ms are spaced logarithmically).
        nearest_m = min(tuned, key=lambda m: abs(math.log(M / m)))
        return tuned[nearest_m]
    return default_config(M, N, K)


def default_config(M: int, N: int, K: int) -> Dict[str, int]:
    """Heuristic configuration for GEMMs without tuned configurations.

    Deliberately conservative so that it runs on every supported device.
    """
    return {
        "BLOCK_SIZE_M": 16 if M <= 16 else 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


def candidate_configs() -> Iterable[Dict[str, int]]:
    """The configuration search space used by ``tune_gemm``.

    Candidates that do not fit the device (e.g. exceed shared memory) are
    skipped during tuning.
    """
    for block_m in (16, 32, 64, 128):
        for block_n in (32, 64, 128):
            for block_k in (32, 64):
                for num_warps in (4, 8):
                    # Wide warps only help for large tiles.
                    if num_warps == 8 and block_m * block_n < 64 * 128:
                        continue
                    for num_stages in (2, 3, 4):
                        yield {
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                            "GROUP_SIZE_M": 8,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }


def tune_gemm(
    N: int,
    K: int,
    Ms: Sequence[int] = DEFAULT_TUNE_MS,
    dtype: torch.dtype = torch.float16,
    save_dir: Optional[Path] = None,
    candidates: Optional[Sequence[Dict[str, int]]] = None,
) -> Path:
    """Benchmark candidate configurations and save the best ones.

    For each ``M`` in ``Ms``, every candidate configuration is benchmarked
    for an ``(M, K) @ (K, N)`` GEMM and the fastest is recorded. The result
    is written as a JSON file named after ``(N, K)`` and the current device.

    ``save_dir`` defaults to this package's ``configs`` directory, so that
    subsequent ``gemm`` calls in the same environment pick up the tuned
    configurations. Kernel authors should pass the ``configs`` directory of
    the kernel *source tree* instead (see ``tune.py`` in the repository
    root) and commit the result, so that the configurations ship with the
    kernel.
    """
    # Import here to avoid a circular import at module load time.
    from .gemm import launch_gemm_kernel

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda"
    if candidates is None:
        candidates = list(candidate_configs())

    best_configs: Dict[int, Dict[str, int]] = {}
    for M in Ms:
        a = torch.randn(M, K, device=device, dtype=dtype)
        b = torch.randn(K, N, device=device, dtype=dtype)
        out = torch.empty(M, N, device=device, dtype=dtype)

        best_time = math.inf
        for config in candidates:
            try:
                time = triton.testing.do_bench(
                    lambda: launch_gemm_kernel(a, b, out, config)
                )
            except triton.runtime.errors.OutOfResources:
                # Configuration does not fit this device, skip it.
                continue
            if time < best_time:
                best_time, best_configs[M] = time, config

        if M not in best_configs:
            raise RuntimeError(f"No candidate configuration fits M={M}, N={N}, K={K}")
        logger.info(
            "Best configuration for M=%d, N=%d, K=%d: %s (%.4f ms)",
            M,
            N,
            K,
            best_configs[M],
            best_time,
        )

    if save_dir is None:
        save_dir = _CONFIGS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / config_file_name(N, K)
    with open(path, "w") as f:
        json.dump({str(m): config for m, config in sorted(best_configs.items())}, f, indent=4)
        f.write("\n")

    # Make sure that new configurations are picked up by subsequent calls.
    _load_tuned_configs.cache_clear()

    return path
