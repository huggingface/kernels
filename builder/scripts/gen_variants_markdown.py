#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
import json
from pathlib import Path

_PLATFORM_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "metal": "Metal",
    "rocm": "ROCm",
    "xpu": "XPU",
}

SPECIFIC_VARIANTS = """# Build variants

A kernel can be compliant for a specific compute framework (e.g. CUDA) or
architecture (e.g. x86_64). For compliance with a compute framework and
architecture combination, all the build variants listed below must be
available. This list will be updated as new PyTorch versions are released.\n
"""

NOARCH_VARIANTS = """## Python-only kernels

Kernels that are in pure Python (e.g. Triton kernels) only need to provide
one or more of the following variants:\n
"""


def json_to_markdown():
    project_root = Path(__file__).parent.parent

    with open(project_root / "build-variants.json", "r") as f:
        data = json.load(f)

    with open(project_root / "docs" / "build-variants.md", "w") as f:
        f.write(SPECIFIC_VARIANTS)
        for arch, platforms in data.items():
            for platform, variants in platforms.items():
                f.write(f"## {_PLATFORM_NAMES[platform]} {arch}\n\n")

                for variant in variants:
                    f.write(f"- `{variant}`\n")

                f.write("\n")
        f.write(NOARCH_VARIANTS)
        backends = { backend for platforms in data.values() for backend in platforms.keys() }
        for backend in sorted(backends):
            f.write(f"- `torch-{backend}`\n")


if __name__ == "__main__":
    json_to_markdown()
