#!/usr/bin/env python3
"""
Download python_depends.json from the kernel-builder repository.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/huggingface/kernel-builder/refs/heads/main/build2cmake/src/python_dependencies.json"
TARGET_DIR = Path(__file__).parent / "src" / "kernels"
TARGET_FILE = TARGET_DIR / "python_depends.json"


def download_json(url: str) -> Dict:
    """Download JSON from URL and return parsed dict."""
    request = Request(url)

    with urlopen(request, timeout=30) as response:
        content = response.read()

    return json.loads(content)


def download_file(url: str, target_path: Path) -> None:
    """Download file from URL and save to target path."""
    data = download_json(url)

    with open(target_path, "w") as f:
        json.dump(data, f, indent=2)


def validate_file(url: str, target_path: Path):
    """Check if local file is up-to-date with remote version.

    Returns True if up-to-date, False otherwise.
    """
    if not target_path.exists():
        return False

    remote_json = download_json(url)

    with open(target_path, "r") as f:
        local_json = json.load(f)

    if local_json != remote_json:
        raise ValueError(f"Local Python dependencies at {target_path} are out of date.")


def main():
    parser = argparse.ArgumentParser(
        description="Download or validate python_depends.json"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that local file is up-to-date instead of downloading",
    )
    args = parser.parse_args()

    if args.validate:
        validate_file(URL, TARGET_FILE)
    else:
        download_file(URL, TARGET_FILE)


if __name__ == "__main__":
    main()
