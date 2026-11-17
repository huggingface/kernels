import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


__all__ = ["relu"]
