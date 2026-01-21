#!/usr/bin/env python3

import argparse
import json
import sys
import gzip
import re
import xml.etree.ElementTree as ET
from typing import Set, Dict
from urllib.parse import urljoin
from urllib.request import urlopen

BASEURL = "https://yum.repos.intel.com/oneapi/"

# XML namespaces used in RPM repo metadata
RPM_NAMESPACES = {
    "common": "http://linux.duke.edu/metadata/common",
    "rpm": "http://linux.duke.edu/metadata/rpm",
}

REPOMD_NAMESPACES = {"repo": "http://linux.duke.edu/metadata/repo"}

VERSION_SUFFIX_RE = re.compile(r"-(\d+\.\d+(\.\d+)?)$")

parser = argparse.ArgumentParser(description="Parse intel oneapi repository")
parser.add_argument("version", help="oneAPI version")


class Package:
    def __init__(self, package_elem, base_url: str):
        self._elem = package_elem
        self._base_url = base_url

        # Parse package metadata.
        name_elem = self._elem.find("common:name", RPM_NAMESPACES)
        self._name = name_elem.text if name_elem is not None else ""

        version_elem = self._elem.find("common:version", RPM_NAMESPACES)
        self._version = version_elem.get("ver", "") if version_elem is not None else ""
        self._release = version_elem.get("rel", "") if version_elem is not None else ""

        arch_elem = self._elem.find("common:arch", RPM_NAMESPACES)
        self._arch = arch_elem.text if arch_elem is not None else ""

        checksum_elem = self._elem.find("common:checksum", RPM_NAMESPACES)
        self._checksum = checksum_elem.text if checksum_elem is not None else ""

        location_elem = self._elem.find("common:location", RPM_NAMESPACES)
        self._location = (
            location_elem.get("href", "") if location_elem is not None else ""
        )

    def __str__(self):
        return f"{self._name} {self._version}"

    def depends(self) -> Set[str]:
        """Extract dependencies, filtering for oneAPI packages"""
        deps = set()

        # Find requires entries in RPM format
        format_elem = self._elem.find("common:format", RPM_NAMESPACES)
        if format_elem is not None:
            requires_elem = format_elem.find("rpm:requires", RPM_NAMESPACES)
            if requires_elem is not None:
                for entry in requires_elem.findall("rpm:entry", RPM_NAMESPACES):
                    dep_name = entry.get("name", "")
                    # Filter out system dependencies and focus on package names
                    if (
                        dep_name
                        and not dep_name.startswith("/")
                        and not dep_name.startswith("rpmlib(")
                    ):
                        deps.add(dep_name)

        return deps

    @property
    def name(self) -> str:
        return self._name

    @property
    def sha256(self) -> str:
        return self._checksum

    @property
    def version(self) -> str:
        version = self._version
        return version

    @property
    def filename(self) -> str:
        return f"{self._name}-{self._version}-{self._release}.{self._arch}.rpm"

    @property
    def url(self) -> str:
        return self._location


def fetch_and_parse_repodata(repo_url: str):
    """Fetch and parse repository metadata"""
    repomd_url = urljoin(repo_url, "repodata/repomd.xml")

    try:
        print(f"Fetching repository metadata from {repomd_url}...", file=sys.stderr)
        with urlopen(repomd_url) as response:
            repomd_content = response.read()

        # Parse repo metadata. From this file we can get the paths to the
        # other metadata files.
        repomd_root = ET.fromstring(repomd_content)

        # Find the primary package metadata.
        primary_location = None
        for data in repomd_root.findall(
            './/repo:data[@type="primary"]', REPOMD_NAMESPACES
        ):
            location_elem = data.find(".//repo:location", REPOMD_NAMESPACES)
            if location_elem is not None:
                primary_location = location_elem.get("href")
                break

        if not primary_location:
            raise Exception("Could not find primary metadata in repomd.xml")

        primary_url = urljoin(repo_url, primary_location)
        print(f"Fetching primary metadata from {primary_url}...", file=sys.stderr)

        with urlopen(primary_url) as response:
            metadata = response.read()

        if primary_location.endswith(".gz"):
            metadata = gzip.decompress(metadata)

        return ET.fromstring(metadata)

    except Exception as e:
        print(f"Error fetching repository metadata: {e}", file=sys.stderr)
        sys.exit(1)


def get_all_packages() -> Dict[str, Package]:
    """Get all packages from the repository"""
    repo_url = BASEURL
    metadata = fetch_and_parse_repodata(repo_url)

    all_packages = {}
    for package_elem in metadata.findall(
        './/common:package[@type="rpm"]', RPM_NAMESPACES
    ):
        pkg = Package(package_elem, repo_url)
        all_packages[pkg.name] = pkg

    return all_packages


def find_target_package(all_packages: Dict[str, Package], version: str) -> Package:
    """Find intel-deep-learning-essentials package with the specified version"""
    target_name = "intel-deep-learning-essentials"

    version_suffix = ".".join(version.split(".")[:2])  # 2025.2.0 -> 2025.2

    # Fallback: Look for version-specific package names
    for name, pkg in all_packages.items():
        if name.startswith(target_name) and name.endswith(f"-{version_suffix}") and (version in pkg.version):
            print(f"Found version match: {name} with version {pkg.version}", file=sys.stderr)
            return pkg

    # If not found, raise an exception
    raise Exception(f"Could not find {target_name} package with version suffix -{version_suffix}")


def resolve_dependencies_recursively(
    target_package: Package,
    all_packages: Dict[str, Package],
    resolved_packages: Dict[str, Package] = None,
) -> Dict[str, Package]:
    """Recursively resolve all dependencies starting from target package"""

    if resolved_packages is None:
        resolved_packages = {}

    # Add the target package if not already added
    if target_package.name not in resolved_packages:
        resolved_packages[target_package.name] = target_package
        print(f"Added package: {target_package.name}", file=sys.stderr)

    # Get dependencies of the current package
    deps = target_package.depends()

    for dep_name in deps:
        # Skip if dependency is already resolved
        if dep_name in resolved_packages:
            continue

        # Find the dependency package in all_packages
        if dep_name in all_packages:
            dep_package = all_packages[dep_name]
            print(
                f"Resolving dependency: {dep_name} for {target_package.name}",
                file=sys.stderr,
            )

            # Recursively resolve this dependency
            resolve_dependencies_recursively(dep_package, all_packages, resolved_packages)
        else:
            print(
                f"Warning: Dependency {dep_name} not found in repository",
                file=sys.stderr,
            )

    return resolved_packages

def main():
    args = parser.parse_args()

    print(f"Fetching all packages from oneAPI repository...", file=sys.stderr)

    # Step 1: Get all packages from repository
    all_packages = get_all_packages()
    print(f"Found {len(all_packages)} total packages in repository", file=sys.stderr)

    # Step 2: Find intel-deep-learning-essentials package with specified version
    try:
        target_package = find_target_package(all_packages, args.version)
        print(
            f"Found target package: {target_package.name} {target_package.version}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Recursively resolve all dependencies
    print(f"Resolving dependencies recursively...", file=sys.stderr)
    required_packages = resolve_dependencies_recursively(target_package, all_packages)

    print(f"Total required packages: {len(required_packages)}", file=sys.stderr)

    # Step 4: Filter dupes like hip-devel vs. hip-devel6.4.1
    filtered_packages = {}
    for name, info in required_packages.items():
        if name.endswith(args.version):
            name_without_version = name[: -len(args.version)]
            if name_without_version not in required_packages:
                filtered_packages[name_without_version] = info
        else:
            filtered_packages[name] = info

    packages = filtered_packages
    print(f"After filtering duplicates: {len(packages)} packages", file=sys.stderr)

    # Step 5: Find -devel packages that should be merged.
    dev_to_merge = {}
    for name in packages.keys():
        base_name = VERSION_SUFFIX_RE.sub("", name)
        if not base_name.endswith("-devel"):
            continue

        match = VERSION_SUFFIX_RE.search(name)
        if match is None:
            if base_name[:-6] in packages:
                dev_to_merge[name] = base_name[:-6]
        else:
            version = match.group(1)
            non_devel_name = f"{base_name[:-6]}-{version}"
            if non_devel_name in packages:
                dev_to_merge[name] = non_devel_name

    print(f"Found {len(dev_to_merge)} packages to merge", file=sys.stderr)

    # Step 6: Generate metadata and merge -devel packages.
    metadata = {}

    # sorted will put -devel after non-devel packages.
    for name in sorted(packages.keys()):
        info = packages[name]
        deps = {
            dev_to_merge.get(dep, dep) for dep in info.depends() if dep in packages
        }

        pkg_metadata = {
            "name": name,
            "sha256": info.sha256,
            "url": urljoin(BASEURL, info.url),
            "version": info.version,
        }

        if name in dev_to_merge:
            target_pkg = dev_to_merge[name]
            if target_pkg not in metadata:
                metadata[target_pkg] = {
                    "deps": set(),
                    "components": [],
                    "version": info.version,
                }
            metadata[target_pkg]["components"].append(pkg_metadata)
            metadata[target_pkg]["deps"].update(deps)
        else:
            metadata[name] = {
                "deps": deps,
                "components": [pkg_metadata],
                "version": info.version,
            }

    # Remove self-references and convert dependencies to list.
    for name, pkg_metadata in metadata.items():
        deps = pkg_metadata["deps"]
        deps -= {name, f"{name}-devel"}
        pkg_metadata["deps"] = list(sorted(deps))


    # Step 7: Filter out unwanted packages by prefix
    unwanted_prefixes = (
        "intel-oneapi-dpcpp-debugger",
    )

    filtered_metadata = {
        name: pkg
        for name, pkg in metadata.items()
        if not any(name.startswith(prefix) for prefix in unwanted_prefixes)
    }

    # Step 8: remove version suffixes from package names.
    filtered_metadata = {VERSION_SUFFIX_RE.sub("", name): pkg for name, pkg in filtered_metadata.items()}
    for pkg in filtered_metadata.values():
        pkg["deps"] = [VERSION_SUFFIX_RE.sub("", dep) for dep in pkg["deps"]]

    print(f"Generated metadata for {len(filtered_metadata)} packages", file=sys.stderr)
    print(json.dumps(filtered_metadata, indent=2))


if __name__ == "__main__":
    main()
