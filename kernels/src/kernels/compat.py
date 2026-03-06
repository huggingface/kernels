import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


try:
    import tvm_ffi

    has_tvm_ffi = True
except ImportError:
    has_tvm_ffi = False

__all__ = ["has_torch", "has_tvm_ffi", "tomllib"]
