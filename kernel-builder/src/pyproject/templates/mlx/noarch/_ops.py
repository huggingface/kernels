def get_backend() -> str:
    """Detect the backend by inspecting MLX."""
    import mlx.core as mx

    if mx.metal.is_available():
        return "metal"
    elif mx.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _find_ops_name() -> str:
    kernel_name = "{{ kernel_name }}"
    unique_id = "{{ kernel_unique_id }}"
    backend = get_backend()
    return f"_{kernel_name}_{backend}_{unique_id}"


_OPS_NAME = _find_ops_name()
