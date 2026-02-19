#!/usr/bin/env python

from typing import Any
from pathlib import Path
import shutil
import sys

from setuptools import setup
from setuptools.command.build import build


class BuildKernel(build):
    """Custom command to build kernel variants."""

    description = "Build kernel variants for different backends"
    user_options = [
        ("backends=", None, "comma-separated list of backends to build (default: all)"),
    ]

    backends: str | None
    parsed_backends: list[str] | None

    def initialize_options(self) -> None:
        super().initialize_options()
        self.backends = None

    def finalize_options(self) -> None:
        super().finalize_options()

    def run(self) -> None:
        """Execute the build command."""
        project_root = Path(__file__).parent

        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        with open(project_root / "build.toml", "rb") as f:
            build_toml: dict[str, Any] = tomllib.load(f)

        backends = set(build_toml["general"]["backends"])
        if self.backends:
            backends.intersection_update(self.backends.split(","))
        backends = list(sorted(backends))

        build_path = project_root / "build"
        build_path.mkdir(parents=True, exist_ok=True)

        print(f"Building kernels for backends: {', '.join(backends)}")
        print(f"Output directory: {build_path.absolute()}")

        for backend in backends:
            self.build_backend(backend, build_path, build_toml)

    def build_backend(
        self, backend: str, build_path: Path, build_toml: dict[str, Any]
    ) -> None:
        """Build kernel variant for a specific backend."""
        variant_dir = build_path / f"torch-{backend}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Building for backend {backend}...")

        project_root = Path(__file__).parent

        kernel_name: str = build_toml["general"]["name"]
        module_name: str = kernel_name.replace("-", "_")

        # Copy over kernel files.
        torch_ext_dir: Path = project_root / "torch-ext" / module_name
        if not torch_ext_dir.exists():
            raise FileNotFoundError(f"torch-ext/{module_name} does not exist")
        for item in torch_ext_dir.rglob("*"):
            if item.is_file():
                if item.suffix == ".pyc" or item.parent.name == "__pycache__":
                    continue
                rel_path = item.relative_to(torch_ext_dir)
                dest = variant_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        # Copy compat module.
        module_dir: Path = variant_dir / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_root / "compat.py", module_dir / "__init__.py")

        # Copy metadata.
        metadata_file: Path = project_root / f"metadata-{backend}.json"
        if not metadata_file.exists():
            raise ValueError(
                f"Metadata file {metadata_file} does not exist, run build2cmake to create it"
            )
        shutil.copy2(metadata_file, variant_dir / f"metadata-{backend}.json")


if __name__ == "__main__":
    setup(
        cmdclass={
            "build_kernel": BuildKernel,
        }
    )
