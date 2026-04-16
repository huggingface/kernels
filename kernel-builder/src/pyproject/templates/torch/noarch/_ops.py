import torch

def get_backend() -> str:
    """Detect the backend by inspecting torch."""
    import torch

    if torch.version.cuda is not None:
        return "cuda"
    elif torch.version.hip is not None:
        return "rocm"
    elif torch.backends.mps.is_available():
        return "metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        return "xpu"
    else:
        return "cpu"


def _find_ops_name() -> str:
    kernel_name = "{{ kernel_name }}"
    unique_id = "{{ kernel_unique_id }}"
    backend = get_backend()
    return f"_{kernel_name}_{backend}_{unique_id}"


_OPS_NAME = _find_ops_name()

ops = getattr(torch.ops, _OPS_NAME)

def add_op_namespace_prefix(op_name: str) -> str:
    """
    Prefix op by namespace.
    """
    return f"{_OPS_NAME}::{op_name}"
