class ReLU:
    has_backward = True
    can_torch_compile = False

    def __init__(self):
        raise AssertionError("Symbol generation must not instantiate layers")


__all__ = ["ReLU"]
