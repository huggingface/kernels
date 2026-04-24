"""
Shared utilities for working with PyTorch versions and wheel URLs.
"""

from packaging.version import Version

PYTHON_VERSION = "cp313"


def cuda_version_to_framework(cuda_version: str) -> str:
    """Convert CUDA version like '11.8' to framework identifier like 'cu118'"""
    return f"cu{cuda_version.replace('.', '')}"


def rocm_version_to_framework(rocm_version: str) -> str:
    """Convert ROCm version like '6.3' to framework identifier like 'rocm6.3'"""
    return f"rocm{rocm_version}"


def system_to_platform(system: str, framework_type: str, torch_version: str) -> str:
    """Convert system identifier to platform string for wheel naming"""
    if framework_type == "xpu":
        xpu_platform_map = {
            "x86_64-linux": "linux_x86_64",
        }
        return xpu_platform_map.get(system, system)

    if system == "aarch64-darwin":
        return (
            "macosx_14_0_arm64"
            if Version(torch_version) >= Version("2.12")
            else "macosx_11_0_arm64"
        )

    platform_map = {
        "x86_64-linux": "manylinux_2_28_x86_64",
        "aarch64-linux": "manylinux_2_28_aarch64",
    }
    return platform_map.get(system, system)


def generate_pytorch_url(
    torch_version: str,
    framework_version: str,
    framework_type: str,
    python_version: str,
    system: str,
    testing: bool = False,
) -> str:
    """Generate PyTorch wheel download URL."""
    platform = system_to_platform(system, framework_type, torch_version)

    if "darwin" in system:
        framework_dir = "cpu"
        version_part = torch_version
        abi_tag = "none" if "2.10" in version_part else python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    elif framework_type == "cpu":
        framework_dir = "cpu"
        version_part = f"{torch_version}%2Bcpu"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    elif framework_type == "xpu":
        framework = "xpu"
        framework_dir = framework
        version_part = f"{torch_version}%2B{framework}"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    else:
        if framework_type == "cuda":
            framework = cuda_version_to_framework(framework_version)
        elif framework_type == "rocm":
            framework = rocm_version_to_framework(framework_version)
        else:
            raise ValueError(f"Unsupported framework type: {framework_type}")

        framework_dir = framework
        version_part = f"{torch_version}%2B{framework}"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"

    test_prefix = "test/" if testing else ""
    return f"https://download.pytorch.org/whl/{test_prefix}{framework_dir}/{wheel_name}"


def generate_pytorch_rc_hf_url(
    torch_version: str,
    framework_version: str,
    framework_type: str,
    python_version: str,
    system: str,
    testing_release: str,
) -> str:
    """Generate PyTorch wheel download URL from HuggingFace for testing releases."""
    platform = system_to_platform(system, framework_type, torch_version)

    if "darwin" in system:
        version_part = torch_version
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    elif framework_type == "cpu":
        version_part = f"{torch_version}%2Bcpu"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    elif framework_type == "xpu":
        framework = "xpu"
        version_part = f"{torch_version}%2B{framework}"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"
    else:
        if framework_type == "cuda":
            framework = cuda_version_to_framework(framework_version)
        elif framework_type == "rocm":
            framework = rocm_version_to_framework(framework_version)
        else:
            raise ValueError(f"Unsupported framework type: {framework_type}")

        version_part = f"{torch_version}%2B{framework}"
        abi_tag = python_version
        wheel_name = f"torch-{version_part}-{python_version}-{abi_tag}-{platform}.whl"

    return f"https://huggingface.co/buckets/danieldk/pytorch-rc/resolve/{testing_release}/{wheel_name}"
