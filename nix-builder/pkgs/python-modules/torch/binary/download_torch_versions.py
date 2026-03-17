#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
"""
Script to download torch wheels from torch-versions.json

This script downloads all the wheel variants specified in torch-versions.json
to a user-provided directory. Wheels that already exist in the target directory
will not be re-downloaded.
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

from torch_versions import PYTHON_VERSION, generate_pytorch_url


def download_wheel(url: str, target_dir: str) -> None:
    """Download a wheel file to the target directory."""
    filename = url.split("/")[-1]
    clean_filename = urllib.parse.unquote(filename)
    target_path = os.path.join(target_dir, clean_filename)

    if os.path.exists(target_path):
        print(f"  Skipping (already exists): {clean_filename}")
        return

    print(f"  Downloading: {clean_filename}")
    print(f"    URL: {url}")

    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"    ✓ Downloaded successfully")
    except Exception as e:
        print(f"    ✗ Error downloading: {e}", file=sys.stderr)
        if os.path.exists(target_path):
            os.remove(target_path)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download PyTorch wheels from torch-versions.json"
    )
    parser.add_argument(
        "torch_versions_file",
        help="Path to torch-versions.json file",
    )
    parser.add_argument("target_dir", help="Directory to download wheels to")
    parser.add_argument(
        "--torch-version",
        help="Only download wheels for this specific torch version (e.g., 2.9.0)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        print(f"Creating target directory: {args.target_dir}")
        os.makedirs(args.target_dir, exist_ok=True)

    try:
        with open(args.torch_versions_file, "r") as f:
            torch_versions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.torch_versions_file} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {args.torch_versions_file}: {e}", file=sys.stderr)
        sys.exit(1)

    total_downloads = 0
    total_skipped = 0

    for entry in torch_versions:
        torch_version = entry.get("torchVersion")
        torch_testing = entry.get("torchTesting")
        cuda_version = entry.get("cudaVersion")
        rocm_version = entry.get("rocmVersion")
        xpu_version = entry.get("xpuVersion")
        cpu = entry.get("cpu", False)
        metal = entry.get("metal", False)
        systems = entry.get("systems", [])

        if not torch_version:
            print(f"Skipping entry without torchVersion: {entry}", file=sys.stderr)
            continue

        if args.torch_version and torch_version != args.torch_version:
            continue

        if cuda_version is not None:
            framework_type = "cuda"
            framework_version = cuda_version
            print(f"Processing torch {torch_version} with CUDA {cuda_version}")
        elif rocm_version is not None:
            framework_type = "rocm"
            framework_version = rocm_version
            print(f"Processing torch {torch_version} with ROCm {rocm_version}")
        elif xpu_version is not None:
            framework_type = "xpu"
            framework_version = xpu_version
            print(f"Processing torch {torch_version} with XPU {xpu_version}")
        elif cpu:
            framework_type = "cpu"
            framework_version = "cpu"
            print(f"Processing torch {torch_version} (CPU build)")
        elif metal:
            framework_type = "cpu"
            framework_version = "cpu"
            print(
                f"Processing torch {torch_version} (CPU-only build with Metal support)"
            )
        else:
            print(
                f"Skipping entry without framework specification: {entry}",
                file=sys.stderr,
            )
            continue

        for system in systems:
            print(f"  Processing system: {system}")

            url = generate_pytorch_url(
                torch_version,
                framework_version,
                framework_type,
                PYTHON_VERSION,
                system,
                testing=torch_testing is not None,
            )

            filename = url.split("/")[-1]
            clean_filename = urllib.parse.unquote(filename)
            target_path = os.path.join(args.target_dir, clean_filename)

            if os.path.exists(target_path):
                total_skipped += 1
            else:
                total_downloads += 1

            try:
                download_wheel(url, args.target_dir)
            except Exception:
                sys.exit(1)

    print(f"  Downloaded: {total_downloads} wheel(s)")
    print(f"  Skipped: {total_skipped} wheel(s)")


if __name__ == "__main__":
    main()
