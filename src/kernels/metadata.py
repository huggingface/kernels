import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Metadata:
    channel: Optional[str]
    python_depends: List[str]

    @staticmethod
    def load_from_variant(variant_path: Path) -> "Metadata":
        metadata_path = variant_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                return Metadata(
                    metadata_dict.get("channel", None),
                    metadata_dict.get("python_depends", []),
                )

        return Metadata(channel=None, python_depends=[])
