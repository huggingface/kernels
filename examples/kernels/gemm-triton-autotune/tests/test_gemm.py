import json

import kernels
import pytest
import torch

gemm_kernel = kernels.get_kernel("kernels-test/gemm-triton-autotune", version=1)

# GEMM output values grow with sqrt(K) for randn inputs, so tolerances are
# scaled by the magnitude of the output values.
DTYPE_TOLERANCES = {
    torch.float16: {"rtol": 2e-2, "atol": 5e-1},
    torch.bfloat16: {"rtol": 4e-2, "atol": 4.0},
}


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 17, 256])
@pytest.mark.parametrize("N,K", [(4096, 4096), (512, 511)])
def test_gemm(device, dtype, M, N, K):
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    ref = (a.float() @ b.float()).to(dtype)
    torch.testing.assert_close(gemm_kernel.gemm(a, b), ref, **DTYPE_TOLERANCES[dtype])

    out = torch.empty(M, N, device=device, dtype=dtype)
    gemm_kernel.gemm(a, b, out=out)
    torch.testing.assert_close(out, ref, **DTYPE_TOLERANCES[dtype])


@pytest.mark.kernels_ci
def test_gemm_validation(device):
    a = torch.randn(4, 8, device=device, dtype=torch.float16)
    with pytest.raises(ValueError, match="Incompatible"):
        gemm_kernel.gemm(a, torch.randn(4, 8, device=device, dtype=torch.float16))
    with pytest.raises(ValueError, match="dtype"):
        gemm_kernel.gemm(a, torch.randn(8, 4, device=device, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="dtype"):
        gemm_kernel.gemm(a.float(), torch.randn(8, 4, device=device))
    with pytest.raises(ValueError, match="2D"):
        gemm_kernel.gemm(a.flatten(), a.flatten())


@pytest.mark.kernels_ci
def test_nearest_tuned_m_is_used(device, monkeypatch, tmp_path):
    tuning = gemm_kernel.tuning

    small = tuning.default_config(16, 128, 128)
    large = tuning.default_config(1024, 128, 128)
    assert small != large
    config_path = tmp_path / tuning.config_file_name(128, 128)
    config_path.write_text(json.dumps({"16": small, "1024": large}))

    monkeypatch.setattr(tuning, "_CONFIGS_DIR", tmp_path)
    tuning._load_tuned_configs.cache_clear()
    try:
        assert tuning.get_config(1, 128, 128) == small
        assert tuning.get_config(32, 128, 128) == small
        assert tuning.get_config(500, 128, 128) == large
        assert tuning.get_config(100000, 128, 128) == large
    finally:
        tuning._load_tuned_configs.cache_clear()


@pytest.mark.kernels_ci
def test_fallback_to_default_config(device):
    tuning = gemm_kernel.tuning
    # No configuration is shipped for this shape, so the heuristic default
    # should be used.
    N, K = 123, 321
    assert tuning.get_config(64, N, K) == tuning.default_config(64, N, K)


@pytest.mark.kernels_ci
def test_shipped_config_is_used(device):
    tuning = gemm_kernel.tuning
    # Configurations tuned on an L4 GPU ship with the kernel (which is also
    # the GPU that CI runs on).
    if tuning.device_name() != "NVIDIA_L4":
        pytest.skip("Shipped configurations were tuned for NVIDIA L4")
    for N, K in [(4096, 4096), (14336, 4096)]:
        assert tuning._load_tuned_configs(N, K) is not None


@pytest.mark.kernels_ci
def test_tune_gemm(device, tmp_path):
    tuning = gemm_kernel.tuning
    candidates = [tuning.default_config(16, 256, 256), tuning.default_config(64, 256, 256)]
    path = gemm_kernel.tune_gemm(N=256, K=256, Ms=(1, 64), save_dir=tmp_path, candidates=candidates)

    assert path.parent == tmp_path
    configs = json.loads(path.read_text())
    assert set(configs.keys()) == {"1", "64"}
    for config in configs.values():
        assert config in candidates
