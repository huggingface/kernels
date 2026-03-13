import logging
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from kernels.backends import CANN, CPU, CUDA, XPU, Metal, Neuron, ROCm
from kernels.variants import _resolve_variants


def _mock_resolve(backend, variants, available):
    with patch("kernels.variants._select_backend") as mock_backend, patch(
        "kernels.variants._build_variants"
    ) as mock_variants:
        mock_backend.return_value = backend
        mock_variants.return_value = variants
        return _resolve_variants(available, None)


class TestResolveVariants:
    def test_exact_match_preferred(self):
        available = {
            "torch28-cxx11-cu129-x86_64-linux",
            "torch28-cxx11-cu128-x86_64-linux",
        }
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux"],
            available,
        )
        assert result == ["torch28-cxx11-cu129-x86_64-linux"]

    def test_cuda_fallback_to_highest_compatible(self):
        available = {
            "torch28-cxx11-cu126-x86_64-linux",
            "torch28-cxx11-cu128-x86_64-linux",
        }
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux"],
            available,
        )
        assert result == ["torch28-cxx11-cu128-x86_64-linux"]

    def test_cuda_fallback_respects_major_version_boundary(self):
        available = {
            "torch28-cxx11-cu118-x86_64-linux",
            "torch28-cxx11-cu126-x86_64-linux",
        }
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux"],
            available,
        )
        assert result == ["torch28-cxx11-cu126-x86_64-linux"]

    def test_cuda_fallback_picks_highest_among_multiple(self):
        available = {
            "torch28-cxx11-cu120-x86_64-linux",
            "torch28-cxx11-cu124-x86_64-linux",
            "torch28-cxx11-cu126-x86_64-linux",
        }
        result = _mock_resolve(
            CUDA(version=Version("12.8")),
            ["torch28-cxx11-cu128-x86_64-linux"],
            available,
        )
        assert result == ["torch28-cxx11-cu126-x86_64-linux"]

    def test_no_compatible_variant_returns_empty(self):
        available = {"torch28-cxx11-cu118-x86_64-linux"}
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux"],
            available,
        )
        assert result == []

    def test_empty_available_returns_empty(self):
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux"],
            set(),
        )
        assert result == []

    def test_multiple_variant_types_resolved(self):
        available = {
            "torch28-cxx11-cu126-x86_64-linux",
            "torch-cuda",
        }
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cxx11-cu129-x86_64-linux", "torch-cuda"],
            available,
        )
        assert "torch-cuda" in result
        assert "torch28-cxx11-cu126-x86_64-linux" in result

    def test_windows_variant_fallback(self):
        available = {"torch28-cu126-x86_64-windows"}
        result = _mock_resolve(
            CUDA(version=Version("12.9")),
            ["torch28-cu129-x86_64-windows"],
            available,
        )
        assert result == ["torch28-cu126-x86_64-windows"]

    @pytest.mark.parametrize(
        "backend, variant",
        [
            (CPU(), "torch28-cpu-x86_64-linux"),
            (Metal(), "torch28-metal-aarch64-darwin"),
            (Neuron(), "torch28-neuron-x86_64-linux"),
            (ROCm(version=Version("6.2")), "torch28-rocm62-x86_64-linux"),
            (XPU(version=Version("2024.2")), "torch28-xpu20242-x86_64-linux"),
            (CANN(version=Version("8.0")), "torch-npu"),
        ],
        ids=["cpu", "metal", "neuron", "rocm", "xpu", "cann"],
    )
    def test_non_cuda_backend_exact_match(self, backend, variant):
        result = _mock_resolve(backend, [variant], {variant})
        assert result == [variant]

    @pytest.mark.parametrize(
        "backend, exact_variant, available_variant",
        [
            (
                ROCm(version=Version("6.2")),
                "torch28-rocm62-x86_64-linux",
                "torch28-rocm61-x86_64-linux",
            ),
            (Neuron(), "torch28-neuron-x86_64-linux", "torch27-neuron-x86_64-linux"),
            (
                XPU(version=Version("2024.2")),
                "torch28-xpu20242-x86_64-linux",
                "torch28-xpu20241-x86_64-linux",
            ),
        ],
        ids=["rocm", "neuron", "xpu"],
    )
    def test_non_cuda_backend_no_fallback(
        self, backend, exact_variant, available_variant
    ):
        result = _mock_resolve(backend, [exact_variant], {available_variant})
        assert result == []


class TestFindKernelInRepoPath:
    def test_logs_when_resolved_to_lower_minor(self, tmp_path, caplog):
        from kernels.utils import _find_kernel_in_repo_path

        exact = "torch28-cxx11-cu129-x86_64-linux"
        resolved = "torch28-cxx11-cu128-x86_64-linux"

        build_dir = tmp_path / "build" / resolved
        build_dir.mkdir(parents=True)
        (build_dir / "__init__.py").touch()

        cuda_129 = CUDA(version=Version("12.9"))
        with (
            patch("kernels.utils._build_variants", return_value=[exact]),
            patch("kernels.utils._select_backend", return_value=cuda_129),
            patch("kernels.variants._select_backend", return_value=cuda_129),
            patch("kernels.variants._build_variants", return_value=[exact]),
            caplog.at_level(logging.INFO),
        ):
            _find_kernel_in_repo_path(tmp_path, "test_kernel")

        assert any(
            "cu129" in r.message and resolved in r.message for r in caplog.records
        )

    def test_no_log_when_exact_match(self, tmp_path, caplog):
        from kernels.utils import _find_kernel_in_repo_path

        exact = "torch28-cxx11-cu129-x86_64-linux"

        build_dir = tmp_path / "build" / exact
        build_dir.mkdir(parents=True)
        (build_dir / "__init__.py").touch()

        cuda_129 = CUDA(version=Version("12.9"))
        with (
            patch("kernels.utils._build_variants", return_value=[exact]),
            patch("kernels.utils._select_backend", return_value=cuda_129),
            patch("kernels.variants._select_backend", return_value=cuda_129),
            caplog.at_level(logging.INFO),
        ):
            _find_kernel_in_repo_path(tmp_path, "test_kernel")

        assert not any("resolved to" in r.message for r in caplog.records)


class TestInstallKernelAllowPatterns:
    def test_downloads_only_resolved_variant(self, tmp_path):
        from kernels.utils import install_kernel

        resolved = "torch28-cxx11-cu128-x86_64-linux"
        exact = "torch28-cxx11-cu129-x86_64-linux"

        build_dir = tmp_path / "build" / resolved
        build_dir.mkdir(parents=True)
        (build_dir / "__init__.py").touch()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(tmp_path)

        cuda_129 = CUDA(version=Version("12.9"))
        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.resolve_status", return_value=("repo/id", "main")),
            patch(
                "kernels.utils._list_available_variants",
                return_value={resolved, "torch28-cxx11-cu126-x86_64-linux"},
            ),
            patch("kernels.utils._build_variants", return_value=[exact]),
            patch("kernels.variants._select_backend", return_value=cuda_129),
            patch("kernels.variants._build_variants", return_value=[exact]),
        ):
            install_kernel("repo/id", revision="main")

        call_kwargs = mock_api.snapshot_download.call_args
        allow_patterns = call_kwargs.kwargs.get("allow_patterns") or call_kwargs[1].get(
            "allow_patterns"
        )
        assert allow_patterns == [f"build/{resolved}/*"]

    def test_downloads_exact_when_available(self, tmp_path):
        from kernels.utils import install_kernel

        exact = "torch28-cxx11-cu129-x86_64-linux"

        build_dir = tmp_path / "build" / exact
        build_dir.mkdir(parents=True)
        (build_dir / "__init__.py").touch()

        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(tmp_path)

        cuda_129 = CUDA(version=Version("12.9"))
        with (
            patch("kernels.utils._get_hf_api", return_value=mock_api),
            patch("kernels.utils.resolve_status", return_value=("repo/id", "main")),
            patch(
                "kernels.utils._list_available_variants",
                return_value={exact, "torch28-cxx11-cu126-x86_64-linux"},
            ),
            patch("kernels.utils._build_variants", return_value=[exact]),
            patch("kernels.variants._select_backend", return_value=cuda_129),
            patch("kernels.variants._build_variants", return_value=[exact]),
        ):
            install_kernel("repo/id", revision="main")

        call_kwargs = mock_api.snapshot_download.call_args
        allow_patterns = call_kwargs.kwargs.get("allow_patterns") or call_kwargs[1].get(
            "allow_patterns"
        )
        assert allow_patterns == [f"build/{exact}/*"]
