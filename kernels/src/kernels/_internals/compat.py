import importlib.util
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


has_sigstore = importlib.util.find_spec("sigstore") is not None
has_mlx = importlib.util.find_spec("mlx") is not None
has_torch = importlib.util.find_spec("torch") is not None
has_tvm_ffi = importlib.util.find_spec("tvm_ffi") is not None


__all__ = ["has_mlx", "has_torch", "has_tvm_ffi", "tomllib"]
