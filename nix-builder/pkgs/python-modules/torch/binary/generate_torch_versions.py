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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

from packaging.version import Version
from torch_versions import (
    PYTHON_VERSION,
    cuda_version_to_framework,
    generate_pytorch_rc_hf_url,
    generate_pytorch_url,
    rocm_version_to_framework,
)

OUTPUT_FILE = "torch-versions-hash.json"


@dataclass
class PendingHash:
    url: str
    version_key: str
    system: str
    framework_key: str
    torch_version: str


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


def compute_nix_hash(url: str) -> Tuple[str, List[str]]:
    """Returns (sri_hash, log_lines). Raises RuntimeError on failure."""
    logs = [f"Fetching hash for: {url}"]
    try:
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
        return convert_result.stdout.strip(), logs
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error computing hash for {url}: {e.stderr}") from e
    except FileNotFoundError as e:
        if "nix-prefetch-url" in str(e):
            raise RuntimeError(
                "nix-prefetch-url not found. Please ensure Nix is installed."
            ) from e
        else:
            raise RuntimeError(
                "nix command not found. Please ensure Nix is installed."
            ) from e


def main():
    parser = argparse.ArgumentParser(
        description="Generate torch-versions-hash.json from torch-versions.json"
    )
    parser.add_argument(
        "torch_versions_file",
        help="Path to torch-versions.json file",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of parallel hash-fetch workers (default: 8)",
    )

    args = parser.parse_args()

    existing_hashes = load_existing_hashes()

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

    pending: List[PendingHash] = []

    # Collect URLs, resolve cache hits immediately.
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

        v = Version(torch_version)
        version_key = f"{v.major}.{v.minor}"

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

            framework_key = framework.replace(".", "")

            if url in existing_hashes:
                hash_value = existing_hashes[url]
                urls_hashes[version_key][system][framework_key] = {
                    "url": url,
                    "hash": hash_value,
                    "version": torch_version,
                }
                print(f"    Hash (cached): {hash_value}")
            else:
                pending.append(
                    PendingHash(
                        url=url,
                        version_key=version_key,
                        system=system,
                        framework_key=framework_key,
                        torch_version=torch_version,
                    )
                )

    # Fetch missing hashes in parallel.
    if pending:
        print(
            f"\nFetching {len(pending)} missing hashes with {args.jobs} parallel worker(s)..."
        )
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(compute_nix_hash, item.url): item for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    hash_value, logs = future.result()
                except RuntimeError as e:
                    print(f"Error: {e}", file=sys.stderr)
                    sys.exit(1)

                for line in logs:
                    print(line)
                print(f"    Hash: {hash_value}")
                urls_hashes[item.version_key][item.system][item.framework_key] = {
                    "url": item.url,
                    "hash": hash_value,
                    "version": item.torch_version,
                }

    try:
        with open(OUTPUT_FILE, "w") as f:
            json.dump(urls_hashes, f, indent=2)
        print(f"Successfully generated {OUTPUT_FILE}")
    except IOError as e:
        print(f"Error writing {OUTPUT_FILE}: {e}", file=sys.stderr)
        sys.exit(1)

    total_urls = sum(
        len(framework_data)
        for version_data in urls_hashes.values()
        for framework_data in version_data.values()
    )
    if total_urls > 0:
        cache_hits = total_urls - len(pending)
        print(
            f"Cache statistics: {cache_hits}/{total_urls} hits ({cache_hits / total_urls * 100:.1f}% hit rate)"
        )


if __name__ == "__main__":
    main()
