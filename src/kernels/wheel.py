import email.policy
import os
from dataclasses import dataclass
from email.message import Message
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

try:
    KERNELS_VERSION = version("kernels")
except PackageNotFoundError:
    KERNELS_VERSION = "unknown"


@dataclass
class Metadata:
    name: str
    version: str
    cuda_version: Optional[str]
    cxx_abi_version: Optional[str]
    torch_version: Optional[str]
    os: Optional[str]
    platform: Optional[str]

    @property
    def is_universal(self) -> bool:
        return self.platform is None


def build_variant_to_wheel(
    repo_id: str,
    *,
    version: str,
    variant_path: Path,
    wheel_dir: Path,
    manylinux_version: str = "2.28",
    python_version: str = "3.9",
) -> Path:
    """
    Create a wheel file from the variant path.
    """
    name = repo_id.split("/")[-1].replace("_", "-")
    metadata = extract_metadata(name, version, variant_path)
    return build_wheel(
        metadata,
        variant_path=variant_path,
        wheel_dir=wheel_dir,
        manylinux_version=manylinux_version,
        python_version=python_version,
    )


def extract_metadata(name: str, version: str, variant_path: Path) -> Metadata:
    """
    Extract metadata from the variant path.
    """
    if variant_path.name == "torch-universal":
        return Metadata(
            name=name,
            version=version,
            cuda_version=None,
            cxx_abi_version=None,
            torch_version=None,
            os=None,
            platform=None,
        )

    if not variant_path.name.startswith("torch"):
        raise ValueError("Currently only conversion of Torch kernels is supported.")

    variant_parts = variant_path.name.removeprefix("torch").split("-")
    if len(variant_parts) != 5:
        raise ValueError(f"Invalid variant name: {variant_path.name}")

    torch_version = f"{variant_parts[0][:-1]}.{variant_parts[0][-1:]}"
    cpp_abi_version = variant_parts[1].removeprefix("cxx")
    cuda_version = variant_parts[2].removeprefix("cu")
    platform = variant_parts[3].replace("-", "_")
    os = variant_parts[4]

    return Metadata(
        name=name,
        version=version,
        cuda_version=cuda_version,
        cxx_abi_version=cpp_abi_version,
        torch_version=torch_version,
        os=os,
        platform=platform,
    )


def build_wheel(
    metadata: Metadata,
    *,
    variant_path: Path,
    wheel_dir: Path,
    manylinux_version: str = "2.28",
    python_version: str = "3.9",
) -> Path:
    """
    Build the wheel file.
    """
    try:
        from wheel.wheelfile import WheelFile  # type: ignore
    except ImportError:
        raise ImportError(
            "The 'wheel' package is required to build wheels. Please install it with: `pip install wheel`"
        )

    name = metadata.name.replace("-", "_")
    python_version_flat = python_version.replace(".", "")

    if metadata.is_universal:
        python_tag = f"py{python_version_flat}"
        abi_tag = "none"
        platform_tag = "any"
        wheel_filename = (
            f"{name}-{metadata.version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        )
        dist_info_dir_name = f"{name}-{metadata.version}.dist-info"
        root_is_purelib = "true"
        requires_dist_torch = "torch"
    else:
        python_tag = f"cp{python_version_flat}"
        abi_tag = "abi3"

        if (
            metadata.torch_version is None
            or metadata.cuda_version is None
            or metadata.cxx_abi_version is None
            or metadata.os is None
            or metadata.platform is None
        ):
            raise ValueError(
                "Torch version, CUDA version, C++ ABI version, OS, and platform must be specified for non-universal wheels."
            )

        local_version = f"torch{metadata.torch_version.replace('.', '')}cu{metadata.cuda_version}cxx{metadata.cxx_abi_version}"

        if metadata.os == "linux":
            platform_tag = (
                f"manylinux_{manylinux_version.replace('.', '_')}_{metadata.platform}"
            )
        else:
            platform_tag = f"{metadata.os}_{metadata.platform.replace('-', '_')}"

        wheel_filename = f"{name}-{metadata.version}+{local_version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
        dist_info_dir_name = f"{name}-{metadata.version}+{local_version}.dist-info"
        root_is_purelib = "false"
        requires_dist_torch = f"torch=={metadata.torch_version}.*"

    wheel_path = wheel_dir / wheel_filename

    wheel_msg = Message(email.policy.compat32)
    wheel_msg.add_header("Wheel-Version", "1.0")
    wheel_msg.add_header("Generator", f"kernels ({KERNELS_VERSION})")
    wheel_msg.add_header("Root-Is-Purelib", root_is_purelib)
    wheel_msg.add_header("Tag", f"{python_tag}-{abi_tag}-{platform_tag}")

    metadata_msg = Message(email.policy.compat32)
    metadata_msg.add_header("Metadata-Version", "2.1")
    metadata_msg.add_header("Name", name)
    metadata_msg.add_header("Version", metadata.version)
    metadata_msg.add_header("Summary", f"{name} kernel")
    metadata_msg.add_header("Requires-Python", ">=3.9")
    metadata_msg.add_header("Requires-Dist", requires_dist_torch)

    source_pkg_dir = variant_path / name

    with WheelFile(wheel_path, "w") as wheel_file:
        for root, dirnames, filenames in os.walk(source_pkg_dir):
            for filename in filenames:
                if filename.endswith(".pyc"):
                    continue

                abs_filepath = os.path.join(root, filename)
                entry_name = os.path.relpath(abs_filepath, variant_path)
                wheel_file.write(abs_filepath, entry_name)

        wheel_metadata_path = os.path.join(dist_info_dir_name, "WHEEL")
        wheel_file.writestr(wheel_metadata_path, str(wheel_msg).encode("utf-8"))

        metadata_path = os.path.join(dist_info_dir_name, "METADATA")
        wheel_file.writestr(metadata_path, str(metadata_msg).encode("utf-8"))

    return wheel_path
