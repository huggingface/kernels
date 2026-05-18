#!/usr/bin/env python3

import argparse
import gzip
import json
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Set
from urllib.parse import urljoin
from urllib.request import urlopen

REPO_URL_TEMPLATES = [
    "http://mirror.transip.net/almalinux/{version}/AppStream/{arch}/os/",
    "http://mirror.transip.net/almalinux/{version}/BaseOS/{arch}/os/",
]

# XML namespaces used in RPM repo metadata
RPM_NAMESPACES = {
    "common": "http://linux.duke.edu/metadata/common",
    "rpm": "http://linux.duke.edu/metadata/rpm",
}

REPOMD_NAMESPACES = {"repo": "http://linux.duke.edu/metadata/repo"}

VERSION_SUFFIX_RE = re.compile(r"-(\d+\.\d+(\.\d+)?)$")


def _rpm_sort_key(s: str) -> str:
    """Zero-pad all digit runs so lexicographic order equals numeric order.

    Works for both RPM version (ver) and release (rel) strings, including
    non-PEP-440 values like '1.0.2o' or '3.el8_10.1'.

    e.g. '1.0.2o'     -> '00000000000000000001.00000000000000000000.00000000000000000002o'
         '3.el8_10.1' -> '00000000000000000003.el00000000000000000008_00000000000000000010.00000000000000000001'
    """
    return re.sub(r"(\d+)", lambda m: m.group(1).zfill(20), s)


parser = argparse.ArgumentParser(description="Parse AlmaLinux repository")
parser.add_argument("--version", default="8", help="AlmaLinux version (default: 8)")
parser.add_argument(
    "--arch", default="x86_64", help="Target architecture (default: x86_64)"
)

TARGET_PACKAGE_NAMES = [
    "gcc-toolset-13",
    "gcc-toolset-14",
]


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
        """Extract required capabilities for this package."""
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

    def provides(self) -> Set[str]:
        """Extract the capabilities/names this package provides."""
        provided = set()

        format_elem = self._elem.find("common:format", RPM_NAMESPACES)
        if format_elem is not None:
            provides_elem = format_elem.find("rpm:provides", RPM_NAMESPACES)
            if provides_elem is not None:
                for entry in provides_elem.findall("rpm:entry", RPM_NAMESPACES):
                    cap = entry.get("name", "")
                    if cap:
                        provided.add(cap)

        return provided

    @property
    def name(self) -> str:
        return self._name

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def release(self) -> str:
        return self._release

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
        return urljoin(self._base_url, self._location)


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


def get_all_packages(arch: str, version: str) -> Dict[str, Package]:
    """Get all packages from the repository"""
    all_packages = {}

    repo_urls = [
        template.format(version=version, arch=arch) for template in REPO_URL_TEMPLATES
    ]
    for repo_url in repo_urls:
        metadata = fetch_and_parse_repodata(repo_url)

        for package_elem in metadata.findall(
            './/common:package[@type="rpm"]', RPM_NAMESPACES
        ):
            package = Package(package_elem, repo_url)

            # Skip packages that don't match the requested architecture.
            # "noarch" packages are architecture-independent and are always included.
            if package.arch not in (arch, "noarch"):
                continue

            # The repo metadata contains multiple versions of the same package. Get
            # them all and then get the latest.

            all_packages.setdefault(package.name, []).append(
                ((package.version, package.release), package)
            )

    packages = {}
    for name, versions in all_packages.items():
        versions.sort(key=lambda x: (_rpm_sort_key(x[0][0]), _rpm_sort_key(x[0][1])))
        packages[name] = versions[-1][1]

    # Build a reverse map from provided capability -> package so that
    # dependencies expressed as .so names or virtual provides can be resolved.
    # When multiple packages provide the same capability, last-write wins;
    # in practice this is rare and the exact provider doesn't matter much.
    provides_map: Dict[str, Package] = {}
    for pkg in packages.values():
        for cap in pkg.provides():
            provides_map[cap] = pkg

    return packages, provides_map


def find_target_packages(all_packages: Dict[str, Package]) -> List[Package]:
    """Find all target packages defined in TARGET_PACKAGE_NAMES."""
    found = []
    for target in TARGET_PACKAGE_NAMES:
        pkg = all_packages.get(target)
        if pkg is not None:
            print(f"Found target package: {pkg.name} {pkg.version}", file=sys.stderr)
            found.append(pkg)
        else:
            raise Exception(f"Could not find target package '{target}' in repository")
    return found


def resolve_dependencies_recursively(
    target_package: Package,
    all_packages: Dict[str, Package],
    provides_map: Dict[str, Package],
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

        # Find the dependency package in all_packages, falling back to the
        # provides map for virtual deps like .so names or capability strings.
        dep_package = all_packages.get(dep_name) or provides_map.get(dep_name)
        if dep_package is not None:
            # dep_name may be a capability (e.g. 'libxcrypt-devel(x86-64)') that
            # resolves to a package already recorded under its real name.  Check
            # the real name too so we don't recurse into an already-resolved
            # package and spin forever on a dependency cycle.
            if dep_package.name in resolved_packages:
                continue

            print(
                f"Resolving dependency: {dep_name} for {target_package.name}",
                file=sys.stderr,
            )

            # Recursively resolve this dependency
            resolve_dependencies_recursively(
                dep_package, all_packages, provides_map, resolved_packages
            )
        else:
            print(
                f"Warning: Dependency {dep_name} not found in repository",
                file=sys.stderr,
            )

    return resolved_packages


def main():
    args = parser.parse_args()

    print("Fetching all packages from AlmaLinux repository...", file=sys.stderr)

    # Step 1: Get all packages from repository
    all_packages, provides_map = get_all_packages(args.arch, args.version)
    print(f"Found {len(all_packages)} total packages in repository", file=sys.stderr)

    # Step 2: Find all target packages
    try:
        target_packages = find_target_packages(all_packages)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Recursively resolve all dependencies, accumulating into one dict
    # so shared dependencies are only included once.
    print("Resolving dependencies recursively...", file=sys.stderr)
    required_packages: Dict[str, Package] = {}
    for target_package in target_packages:
        resolve_dependencies_recursively(
            target_package, all_packages, provides_map, required_packages
        )

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
        deps = set()
        for dep in info.depends():
            if dep in packages:
                deps.add(dev_to_merge.get(dep, dep))
            elif dep in provides_map:
                # dep is a capability (e.g. 'libmpc.so.3()(64bit)'); use the
                # name of the package that provides it instead.
                resolved = provides_map[dep].name
                deps.add(dev_to_merge.get(resolved, resolved))

        pkg_metadata = {
            "name": name,
            "sha256": info.sha256,
            "url": info.url,
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
    unwanted_prefixes = ("intel-oneapi-dpcpp-debugger",)

    filtered_metadata = {
        name: pkg
        for name, pkg in metadata.items()
        if not any(name.startswith(prefix) for prefix in unwanted_prefixes)
    }

    # Step 8: remove version suffixes from package names.
    filtered_metadata = {
        VERSION_SUFFIX_RE.sub("", name): pkg for name, pkg in filtered_metadata.items()
    }
    for pkg in filtered_metadata.values():
        pkg["deps"] = [VERSION_SUFFIX_RE.sub("", dep) for dep in pkg["deps"]]

    # Step 9: Rename any occurrence of 'c++' -> 'cxx' in package names so they
    # are valid Nix attribute names (++ is an operator in the Nix expression
    # language). This covers e.g. libstdc++ -> libstdcxx, gcc-c++ -> gcc-cxx.
    def _cxx(s: str) -> str:
        return s.replace("c++", "cxx")

    filtered_metadata = {
        _cxx(name): {
            **pkg,
            "deps": [_cxx(d) for d in pkg["deps"]],
            "components": [{**c, "name": _cxx(c["name"])} for c in pkg["components"]],
        }
        for name, pkg in filtered_metadata.items()
    }

    print(f"Generated metadata for {len(filtered_metadata)} packages", file=sys.stderr)
    print(json.dumps(filtered_metadata, indent=2))


if __name__ == "__main__":
    main()
