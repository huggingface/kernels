import importlib
import platform
import sys

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from packaging.version import parse

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def build_variant():
    torch_version = parse(torch.__version__)
    cuda_version = parse(torch.version.cuda)
    cxxabi = "cxx11" if torch.compiled_with_cxx11_abi() else "cxx98"
    cpu = platform.machine()
    os = platform.system().lower()

    return f"torch{torch_version.major}{torch_version.minor}-{cxxabi}-cu{cuda_version.major}{cuda_version.minor}-{cpu}-{os}"


def import_from_path(module_name: str, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_package_path(repo_id: str):
    repo_path = snapshot_download(repo_id, allow_patterns=f"build/{build_variant()}/*")
    return f"{repo_path}/build/{build_variant()}"


def get_metadata(repo_id: str):
    with open(hf_hub_download(repo_id, "build.toml"), "rb") as f:
        return tomllib.load(f)


def get_kernel(repo_id: str):
    package_name = get_metadata(repo_id)["torch"]["name"]
    package_path = get_package_path(repo_id)
    return import_from_path(package_name, f"{package_path}/{package_name}/__init__.py")
