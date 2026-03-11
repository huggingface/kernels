from packaging.version import Version

from kernels.backends import CANN, CPU, CUDA, XPU, Metal, Neuron, ROCm
from kernels.variants import _compatible_backend_variants


class TestCompatibleBackendVariants:
    # Only CUDA for now.

    def test_cuda_single_variant_minor_zero(self):
        backend = CUDA(version=Version("12.0"))
        assert _compatible_backend_variants(backend) == ["cu120"]

    def test_cuda_returns_descending_minor_versions(self):
        backend = CUDA(version=Version("12.9"))
        variants = _compatible_backend_variants(backend)
        assert variants == [f"cu12{i}" for i in range(9, -1, -1)]

    def test_cuda_first_variant_is_exact_match(self):
        backend = CUDA(version=Version("12.6"))
        variants = _compatible_backend_variants(backend)
        assert variants[0] == "cu126"

    def test_cuda_last_variant_is_major_minor_zero(self):
        backend = CUDA(version=Version("12.6"))
        variants = _compatible_backend_variants(backend)
        assert variants[-1] == "cu120"

    def test_cuda_resolves_to_largest_available_minor(self):
        # System has CUDA 12.9; available builds: 12.6, 12.8, 13.0.
        # Expected: resolve to 12.8 (largest z <= 9 that is available).
        backend = CUDA(version=Version("12.9"))
        candidates = _compatible_backend_variants(backend)
        available = {"cu126", "cu128", "cu130"}
        resolved = next((v for v in candidates if v in available), None)
        assert resolved == "cu128"

    def test_cuda_falls_back_when_exact_not_available(self):
        # System has CUDA 12.9; only 12.6 available for major 12.
        backend = CUDA(version=Version("12.9"))
        candidates = _compatible_backend_variants(backend)
        available = {"cu126"}
        resolved = next((v for v in candidates if v in available), None)
        assert resolved == "cu126"

    def test_cuda_no_match_when_only_higher_minor_available(self):
        # System has CUDA 12.6; only 12.8 available — should not match.
        backend = CUDA(version=Version("12.6"))
        candidates = _compatible_backend_variants(backend)
        available = {"cu128"}
        resolved = next((v for v in candidates if v in available), None)
        assert resolved is None

    def test_cuda_no_match_for_different_major(self):
        # System has CUDA 12.9; only 13.x builds available — should not match.
        backend = CUDA(version=Version("12.9"))
        candidates = _compatible_backend_variants(backend)
        available = {"cu130", "cu131"}
        resolved = next((v for v in candidates if v in available), None)
        assert resolved is None

    def test_cuda_major_version_preserved(self):
        backend = CUDA(version=Version("11.8"))
        variants = _compatible_backend_variants(backend)
        assert all(v.startswith("cu11") for v in variants)

    # Non-CUDA

    def test_cpu_returns_single_variant(self):
        assert _compatible_backend_variants(CPU()) == ["cpu"]

    def test_metal_returns_single_variant(self):
        assert _compatible_backend_variants(Metal()) == ["metal"]

    def test_neuron_returns_single_variant(self):
        assert _compatible_backend_variants(Neuron()) == ["neuron"]

    def test_rocm_returns_single_variant(self):
        backend = ROCm(version=Version("6.2"))
        assert _compatible_backend_variants(backend) == ["rocm62"]

    def test_xpu_returns_single_variant(self):
        backend = XPU(version=Version("2024.2"))
        assert _compatible_backend_variants(backend) == [backend.variant]

    def test_cann_returns_single_variant(self):
        backend = CANN(version=Version("8.0"))
        assert _compatible_backend_variants(backend) == ["cann80"]
