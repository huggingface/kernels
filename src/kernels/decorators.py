from typing import TYPE_CHECKING

from .utils import get_kernel

if TYPE_CHECKING:
    from torch import nn


def use_hub_kernel(
    repo_id: str,
    *,
    layer_name: str,
    revision: str = "main",
    fallback_on_error: bool = True,
):
    """
    Replace a layer with a layer from the kernel hub.

    When `fallback_on_error` is True, the original layer will be used if
    the kernel's layer could not be loaded.
    """

    def decorator(cls):
        try:
            return _get_kernel_layer(
                repo_id=repo_id, layer_name=layer_name, revision=revision
            )
        except Exception as e:
            if not fallback_on_error:
                raise e

        return cls

    return decorator


def _get_kernel_layer(*, repo_id: str, layer_name: str, revision: str) -> "nn.Module":
    """Get a layer from a kernel."""

    from torch import nn

    kernel = get_kernel(repo_id, revision=revision)

    if getattr(kernel, "layers", None) is None:
        raise ValueError(
            f"Kernel `{repo_id}` at revision `{revision}` does not define any layers."
        )

    layer = getattr(kernel.layers, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{layer_name}` not found in kernel `{repo_id}`.")
    if not issubclass(layer, nn.Module):
        raise TypeError(f"Layer `{layer_name}` is not a Torch layer.")
    return layer
