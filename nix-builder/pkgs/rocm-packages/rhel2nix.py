#!/usr/bin/env python3

import argparse
import json
import re
import sys
import gzip
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin
from urllib.request import urlopen

BASEURL = "https://repo.amd.com/rocm/packages-multi-arch/rhel{rhel_version}/{arch}/"

RHEL_VERSIONS = ["8", "9"]
ARCHES = ["x86_64"]

# XML namespaces used in RPM repo metadata
RPM_NAMESPACES = {
    "common": "http://linux.duke.edu/metadata/common",
    "rpm": "http://linux.duke.edu/metadata/rpm",
}

REPOMD_NAMESPACES = {"repo": "http://linux.duke.edu/metadata/repo"}

# Suffix marking per-GPU-target subpackages, e.g. amdrocm-blas7.14-gfx942.
GFX_SUFFIX = re.compile(r"-gfx\d+[a-z0-9]*$")

# Pure dependency-tier metapackages. Not emitted; their dependencies are
# spliced through to dependents.
TRANSPARENT_BUNDLES = {
    "amdrocm-core",
    "amdrocm-core-devel",
    "amdrocm-core-sdk",
    "amdrocm-hpc",
    "amdrocm-hpc-sdk",
}

# Groups of bundles with genuine link-level cyclic dependencies (proven from
# soname requires in primary.xml; see HANDOFF.md "Known problem"). nix
# buildInputs cannot express a cycle, so each group is unioned into a single
# manifest entry, keyed by its first (canonical) member, before dependency
# resolution runs. Internal cross-references between merged members then
# collapse to self-edges and are dropped like any other self-dependency.
MERGE_GROUPS: List[List[str]] = [
    ["amdrocm-runtime", "amdrocm-base", "amdrocm-llvm"],
    ["amdrocm-blas", "amdrocm-solver", "amdrocm-sparse"],
]

parser = argparse.ArgumentParser(description="Parse ROCm RHEL repository")
parser.add_argument("version", help="ROCm version, e.g. 7.14.1")
parser.add_argument(
    "--rhel-version", help="RHEL version", choices=RHEL_VERSIONS, default="8"
)
parser.add_argument(
    "--arch", help="Repository architecture", choices=ARCHES, default="x86_64"
)


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

    def requires(self) -> List[str]:
        """Raw requires entries from the RPM metadata."""
        deps = []
        format_elem = self._elem.find("common:format", RPM_NAMESPACES)
        if format_elem is not None:
            requires_elem = format_elem.find("rpm:requires", RPM_NAMESPACES)
            if requires_elem is not None:
                for entry in requires_elem.findall("rpm:entry", RPM_NAMESPACES):
                    deps.append(entry.get("name", ""))
        return deps

    @property
    def name(self) -> str:
        return self._name

    @property
    def sha256(self) -> str:
        return self._checksum

    @property
    def version(self) -> str:
        return self._version

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


def package_info(*, rhel_version: str, arch: str):
    """Generator that yields Package objects from the RHEL repository"""
    repo_url = BASEURL.format(rhel_version=rhel_version, arch=arch)

    metadata = fetch_and_parse_repodata(repo_url)

    # Iterate through all packages in the metadata
    for package_elem in metadata.findall(
        './/common:package[@type="rpm"]', RPM_NAMESPACES
    ):
        yield Package(package_elem, repo_url)


def major_minor(version: str) -> str:
    """'7.14.1' -> '7.14' (the suffix used in versioned package names)."""
    return ".".join(version.split(".")[:2])


def __main__():
    args = parser.parse_args()
    version_mm = major_minor(args.version)

    print(
        f"Fetching ROCm {args.version} packages for RHEL {args.rhel_version} ({args.arch})...",
        file=sys.stderr,
    )

    # Select the packages that belong to the requested ROCm version. The
    # multi-arch repo serves several versions from one URL.
    packages = {}
    available_versions = set()
    for pkg in package_info(rhel_version=args.rhel_version, arch=args.arch):
        available_versions.add(pkg.version)
        if pkg.version != args.version:
            continue
        if "debuginfo" in pkg.name or "debugsource" in pkg.name:
            continue
        packages[pkg.name] = pkg

    if not packages:
        print(
            f"No packages found for version {args.version}. "
            f"Available versions: {', '.join(sorted(available_versions))}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(packages)} packages", file=sys.stderr)

    def strip_suffixes(name: str) -> str:
        """Reduce a package name to its bundle base.

        amdrocm-blas7.14-gfx942 -> amdrocm-blas7.14 -> amdrocm-blas
        """
        base = GFX_SUFFIX.sub("", name)
        if base.endswith(version_mm):
            base = base[: -len(version_mm)]
        return base

    # Drop unversioned alias packages (amdrocm-blas -> amdrocm-blas7.14,
    # amdrocm-core-gfx942 -> amdrocm-core7.14-gfx942): the unversioned
    # names are 1-file aliases; the versioned names are real.
    def is_alias(name: str) -> bool:
        base = GFX_SUFFIX.sub("", name)
        if base.endswith(version_mm):
            return False
        return f"{base}{version_mm}" in packages

    aliases = [name for name in packages if is_alias(name)]
    for name in aliases:
        del packages[name]
    print(f"Dropped {len(aliases)} unversioned alias packages", file=sys.stderr)

    # Group packages into bundles.
    bundles: Dict[str, List[Package]] = {}
    for pkg in packages.values():
        bundles.setdefault(strip_suffixes(pkg.name), []).append(pkg)

    # Merge variant bundles into their base bundle: "-host" always (the
    # base package is just a meta), "-devel" when the base exists.
    for bundle in list(bundles):
        target = None
        if bundle.endswith("-host"):
            target = bundle[: -len("-host")]
        elif bundle.endswith("-devel") and bundle[: -len("-devel")] in bundles:
            target = bundle[: -len("-devel")]
        if target is not None and target in bundles:
            bundles[target].extend(bundles.pop(bundle))

    print(f"Grouped into {len(bundles)} bundles", file=sys.stderr)

    # Merge cyclic bundle groups (see MERGE_GROUPS above) into their
    # canonical member, so the cycle never shows up in the manifest.
    merge_target: Dict[str, str] = {}
    for group in MERGE_GROUPS:
        canonical = group[0]
        for name in group:
            merge_target[name] = canonical
        if canonical not in bundles:
            continue
        for name in group[1:]:
            if name in bundles:
                bundles[canonical].extend(bundles.pop(name))
    print(
        f"Merged cyclic bundle groups, {len(bundles)} bundles remain",
        file=sys.stderr,
    )

    def normalize_dep(name: str) -> Optional[str]:
        """Normalize a requires entry to a bundle name, or None to drop."""
        if not name or "(" in name or name.startswith("/"):
            # System, soname, rpmlib and config() dependencies.
            return None
        base = strip_suffixes(name)
        # Mirror the variant merging above.
        if base.endswith("-host") and base[: -len("-host")] in bundles:
            base = base[: -len("-host")]
        elif base.endswith("-devel") and base[: -len("-devel")] in bundles:
            base = base[: -len("-devel")]
        # Redirect merged-away cyclic group members to their canonical name.
        return merge_target.get(base, base)

    def dep_bundles(bundle: str, seen: Set[str]) -> Set[str]:
        """Direct dependency bundles of a bundle, splicing through
        transparent metapackage tiers."""
        deps = set()
        for pkg in bundles.get(bundle, []):
            for raw in pkg.requires():
                dep = normalize_dep(raw)
                if not dep or dep == bundle or dep in seen:
                    continue
                if dep in TRANSPARENT_BUNDLES:
                    deps |= dep_bundles(dep, seen | {bundle})
                elif dep in bundles:
                    deps.add(dep)
                # Otherwise: a system dependency, not part of the repo.
        return deps

    # Build the manifest, one entry per bundle.
    metadata = {}
    for bundle in sorted(bundles):
        if bundle in TRANSPARENT_BUNDLES:
            continue
        components = [
            {
                "name": pkg.name,
                "sha256": pkg.sha256,
                "url": pkg.url,
                "version": pkg.version,
            }
            for pkg in sorted(bundles[bundle], key=lambda p: p.name)
        ]

        # dep_bundles already splices through (and drops) transparent tiers.
        deps = dep_bundles(bundle, {bundle})

        metadata[bundle] = {
            "deps": deps,
            "components": components,
            "version": components[0]["version"] if components else args.version,
        }

    # Resolve dependency cycles (the new repo has them at the
    # shared-library level, e.g. amdrocm-base -> amdrocm-runtime ->
    # amdrocm-llvm -> amdrocm-runtime); nix buildInputs cannot be
    # cyclic. Cyclic edges are dropped with a warning; see HANDOFF.md
    # for the follow-up fix.
    graph: Dict[str, List[str]] = {key: [] for key in metadata}

    def reaches(src: str, dst: str) -> bool:
        stack, visited = [src], set()
        while stack:
            node = stack.pop()
            if node == dst:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(graph.get(node, []))
        return False

    for key in sorted(metadata):
        kept = []
        for dep in sorted(metadata[key]["deps"]):
            if reaches(dep, key):
                print(f"Breaking dependency cycle: {key} -> {dep}", file=sys.stderr)
            else:
                graph[key].append(dep)
                kept.append(dep)
        metadata[key]["deps"] = kept

    print(f"Generated metadata for {len(metadata)} packages", file=sys.stderr)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    __main__()
