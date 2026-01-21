#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3
"""
Script to generate torch-versions-hash.json from torch-versions.json

This script downloads all the variants that are specified and computes
their Nix store hashes. Variants for which the hash was already computed
will not be processed again to avoid redownloading/hashing.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
from typing import Dict

from torch_versions import (
    PYTHON_VERSION,
    cuda_version_to_framework,
    generate_pytorch_rc_hf_url,
    generate_pytorch_url,
    rocm_version_to_framework,
    version_to_major_minor,
)

OUTPUT_FILE = "torch-versions-hash.json"


def load_existing_hashes() -> Dict[str, str]:
    """Load existing URL -> hash mappings from output file"""
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r") as f:
                data = json.load(f)
                url_to_hash = {}
                for version_data in data.values():
                    for system_data in version_data.values():
                        for framework_data in system_data.values():
                            if (
                                isinstance(framework_data, dict)
                                and "url" in framework_data
                                and "hash" in framework_data
                            ):
                                if framework_data["hash"]:
                                    url_to_hash[framework_data["url"]] = framework_data[
                                        "hash"
                                    ]
                return url_to_hash
        except (json.JSONDecodeError, IOError) as e:
            # If we fail to parse the file, emit a warning and start from scratch.
            print(
                f"Warning: Could not load existing {OUTPUT_FILE}: {e}", file=sys.stderr
            )
    return {}


def compute_nix_hash(url: str) -> str:
    try:
        print(f"Fetching hash for: {url}")

        # Some URL encodings are not valid in store paths, so unquote.
        filename = url.split("/")[-1]
        clean_filename = urllib.parse.unquote(filename)

        result = subprocess.run(
            ["nix-prefetch-url", "--type", "sha256", "--name", clean_filename, url],
            check=True,
            capture_output=True,
            text=True,
        )
        base32_hash = result.stdout.strip()

        # Convert base32 hash to SRI format.
        convert_result = subprocess.run(
            [
                "nix",
                "hash",
                "convert",
                "--hash-algo",
                "sha256",
                "--from",
                "nix32",
                base32_hash,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return convert_result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error computing hash for {url}: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        if "nix-prefetch-url" in str(e):
            print(
                "Error: nix-prefetch-url not found. Please ensure Nix is installed.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(
                "Error: nix command not found. Please ensure Nix is installed.",
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate torch-versions-hash.json from torch-versions.json"
    )
    parser.add_argument(
        "torch_versions_file",
        help="Path to torch-versions.json file",
    )

    args = parser.parse_args()

    existing_hashes = load_existing_hashes()
    cache_hits = 0
    cache_misses = 0

    try:
        with open(args.torch_versions_file, "r") as f:
            torch_versions = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.torch_versions_file} not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing {args.torch_versions_file}: {e}", file=sys.stderr)
        sys.exit(1)

    urls_hashes = {}

    print(f"Processing {len(torch_versions)} entries from {args.torch_versions_file}")
    print(f"Found {len(existing_hashes)} existing hashes")

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

        version_key = version_to_major_minor(torch_version)

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

        if version_key not in urls_hashes:
            urls_hashes[version_key] = {}

        for system in systems:
            print(f"  Processing system: {system}")

            if system not in urls_hashes[version_key]:
                urls_hashes[version_key][system] = {}

            if "darwin" in system:
                framework = "cpu"
            else:
                if framework_type == "cuda":
                    framework = cuda_version_to_framework(framework_version)
                elif framework_type == "rocm":
                    framework = rocm_version_to_framework(framework_version)
                elif framework_type == "xpu":
                    framework = "xpu"
                elif framework_type == "cpu":
                    framework = "cpu"
                else:
                    print(
                        f"    ⚠️  Warning: Unknown framework type {framework_type} for Linux system {system}",
                        file=sys.stderr,
                    )
                    continue

            if torch_testing is not None:
                url = generate_pytorch_rc_hf_url(
                    torch_version,
                    framework_version,
                    framework_type,
                    PYTHON_VERSION,
                    system,
                    torch_testing,
                )
            else:
                url = generate_pytorch_url(
                    torch_version,
                    framework_version,
                    framework_type,
                    PYTHON_VERSION,
                    system,
                    testing=False,
                )
            print(f"    URL: {url}")

            was_cached = url in existing_hashes
            if was_cached:
                hash_value = existing_hashes[url]
            else:
                hash_value = compute_nix_hash(url)

            if was_cached:
                cache_hits += 1
            else:
                cache_misses += 1

            urls_hashes[version_key][system][framework.replace(".", "")] = {
                "url": url,
                "hash": hash_value,
                "version": torch_version,
            }

            print(f"    Hash: {hash_value}")

    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(urls_hashes, f, indent=2)
        print(f"Successfully generated {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    total_urls = cache_hits + cache_misses
    if total_urls > 0:
        print(
            f"Cache statistics: {cache_hits}/{total_urls} hits ({cache_hits / total_urls * 100:.1f}% hit rate)"
        )


if __name__ == "__main__":
    main()
