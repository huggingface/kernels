import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Metadata:
    python_depends: List[str]
    version: Optional[int]

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
