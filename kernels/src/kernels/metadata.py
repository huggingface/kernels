import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub.dataclasses import strict


@strict
@dataclass
class Metadata:
    id: str | None
    python_depends: list[str]
    version: int | None

    @staticmethod
    def load_from_variant(variant_path: Path) -> "Metadata":
        metadata_path = variant_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata_dict = json.load(f)
                if (kernel_id := metadata_dict.get("id", None)) is None:
                    warnings.warn(
                        f"Metadata for kernel loaded from `{variant_path}` does have an identifier,"
                        " identifiers will become required in kernels >= 0.15\n"
                        "Run `nix flake update in your kernel directory and rebuild to generate metadata.",
                        UserWarning,
                        stacklevel=2,
                    )
                return Metadata(
                    id=kernel_id,
                    python_depends=metadata_dict.get("python-depends", []),
                    version=metadata_dict.get("version", None),
                )

        warnings.warn(
            f"Kernel loaded from `{variant_path}` does not have metadata,"
            " metadata will be required in kernels >= 0.15\n"
            "Run `nix flake update in your kernel directory and rebuild to generate metadata.",
            UserWarning,
            stacklevel=2,
        )

        return Metadata(id=None, version=None, python_depends=[])
