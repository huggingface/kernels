#!/usr/bin/env python3

import argparse
import json
import sys
import gzip
import xml.etree.ElementTree as ET
from typing import Set
from urllib.parse import urljoin
from urllib.request import urlopen

BASEURL = "https://repo.radeon.com/rocm/rhel{rhel_version}/{version}/main/"

RHEL_VERSIONS = ["8", "9"]

# XML namespaces used in RPM repo metadata
RPM_NAMESPACES = {
    "common": "http://linux.duke.edu/metadata/common",
    "rpm": "http://linux.duke.edu/metadata/rpm",
}

REPOMD_NAMESPACES = {"repo": "http://linux.duke.edu/metadata/repo"}

parser = argparse.ArgumentParser(description="Parse ROCm RHEL repository")
parser.add_argument("version", help="ROCm version")
parser.add_argument(
    "--rhel-version", help="RHEL version", choices=RHEL_VERSIONS, default="8"
)


class Package:
    def __init__(self, package_elem, rhel_version: str, base_url: str):
        self._elem = package_elem
        self._rhel_version = rhel_version
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

    def depends(self, version: str) -> Set[str]:
        """Extract dependencies, filtering for ROCm packages"""
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
                        # Remove any version suffix that matches the ROCm version
                        if dep_name.endswith(version):
                            dep_name = dep_name[: -len(version)]
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
        # Remove RHEL-specific version suffix (e.g., .el8)
        version = self._version
        if f".el{self._rhel_version}" in version:
            version = version.split(f".el{self._rhel_version}")[0]
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


def package_info(*, rhel_version: str, version: str):
    """Generator that yields Package objects from the RHEL repository"""
    repo_url = BASEURL.format(rhel_version=rhel_version, version=version)

    metadata = fetch_and_parse_repodata(repo_url)

    # Iterate through all packages in the metadata
    for package_elem in metadata.findall(
        './/common:package[@type="rpm"]', RPM_NAMESPACES
    ):
        yield Package(package_elem, rhel_version, repo_url)


def __main__():
    args = parser.parse_args()
    packages = {}

    print(
        f"Fetching ROCm {args.version} packages for RHEL {args.rhel_version}...",
        file=sys.stderr,
    )

    for pkg in package_info(rhel_version=args.rhel_version, version=args.version):
        if "debuginfo" not in pkg.name and "debugsource" not in pkg.name:
            packages[pkg.name] = pkg

    print(f"Found {len(packages)} packages", file=sys.stderr)

    filtered_packages = {}
    # Filter dupes like hip-devel vs. hip-devel6.4.1
    for name, info in packages.items():
        if name.endswith(args.version):
            name_without_version = name[: -len(args.version)]
            if name_without_version not in packages:
                filtered_packages[name_without_version] = info
        else:
            filtered_packages[name] = info
    packages = filtered_packages

    print(f"After filtering duplicates: {len(packages)} packages", file=sys.stderr)

    # First pass: Find -devel and -rpath packages that should be merged.
    dev_to_merge = {}
    for name in packages.keys():
        if name.endswith("-devel") and name[:-6] in packages:
            dev_to_merge[name] = name[:-6]
        elif name.endswith("-devel-rpath") and name[:-12] in packages:
            dev_to_merge[name] = name[:-12]
        elif name.endswith("-rpath") and name[:-6] in packages:
            dev_to_merge[name] = name[:-6]

    print(f"Found {len(dev_to_merge)} packages to merge", file=sys.stderr)

    # Second pass: get ROCm dependencies and merge -devel packages.
    metadata = {}

    # sorted will put -devel after non-devel packages.
    for name in sorted(packages.keys()):
        info = packages[name]
        deps = {
            dev_to_merge.get(dep, dep)
            for dep in info.depends(args.version)
            if dep in packages
        }

        pkg_metadata = {
            "name": name,
            "sha256": info.sha256,
            "url": urljoin(
                BASEURL.format(rhel_version=args.rhel_version, version=args.version),
                info.url,
            ),
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
        deps -= {name, f"{name}-rpath"}
        pkg_metadata["deps"] = list(sorted(deps))

    print(f"Generated metadata for {len(metadata)} packages", file=sys.stderr)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    __main__()
