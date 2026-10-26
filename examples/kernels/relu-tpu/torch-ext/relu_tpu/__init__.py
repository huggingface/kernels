import jax
import jax.numpy as jnp
from torch_tpu._internal.pallas import jax_op


def _jax_relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(x, 0)


relu = jax_op("relu_tpu::relu", _jax_relu)

from . import layers  # noqa: E402

__all__ = ["layers", "relu"]
