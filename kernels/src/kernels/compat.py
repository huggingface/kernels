import importlib.util
import tomllib

has_sigstore = importlib.util.find_spec("sigstore") is not None
has_torch = importlib.util.find_spec("torch") is not None
has_tvm_ffi = importlib.util.find_spec("tvm_ffi") is not None


__all__ = ["has_torch", "has_tvm_ffi", "tomllib"]
