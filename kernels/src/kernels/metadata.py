import json
from dataclasses import dataclass
from pathlib import Path

from kernels.compat import tomllib


@dataclass
class Metadata:
    python_depends: list[str]
    version: int | None

    @staticmethod
    def load_from_build_toml(build_toml_path: Path) -> "Metadata":
        if build_toml_path.exists():
            with open(build_toml_path, "rb") as f:
                data = tomllib.load(f)
                version = data.get("general", {}).get("version", None)
                return Metadata(
                    version=version,
                    python_depends=[],
                )

        return Metadata(version=None, python_depends=[])

    @staticmethod
    def load_from_variant(variant_path: Path) -> "Metadata":
        metadata_path = variant_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                return Metadata(
                    python_depends=metadata_dict.get("python-depends", []),
                    version=metadata_dict.get("version", None),
                )

        return Metadata(version=None, python_depends=[])
