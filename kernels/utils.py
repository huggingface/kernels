import torch
import sys
import importlib

from pathlib import Path
from packaging.version import parse
from huggingface_hub import hf_hub_download, snapshot_download


def torch_version():
    return parse(torch.__version__)


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_kernel(repo_id: str):
    lib = snapshot_download(repo_id,
                            allow_patterns=f"build/{torch_version()}/*.so")
    sys.path.append(lib + f"/build/{torch_version()}")
    api = hf_hub_download(repo_id, filename="api.py")

    return import_from_path("api", api)



