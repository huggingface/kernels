import inspect
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict

from .utils import get_kernel

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class Device:
    type: str

    # In the future we might add compute capabilities, etc.

    def __eq__(self, other):
        return isinstance(other, Device) and self.type == other.type

    def __hash__(self):
        return hash(self.type)


@dataclass
class LayerRepository:
    """
    Repository and name of a layer.
    """

    layer_name: str = field(
        metadata={"help": "The name of the layer in the kernel repository."}
    )
    repo_id: str = field(metadata={"help": "The kernel hub repository with the layer."})
    revision: str = field(
        default="main", metadata={"help": "The revision of the layer."}
    )

    def __eq__(self, other):
        return (
            isinstance(other, LayerRepository)
            and self.layer_name == other.layer_name
            and self.repo_id == other.repo_id
            and self.revision == other.revision
        )

    def __hash__(self):
        return hash((self.layer_name, self.repo_id, self.revision))


_KERNEL_MAPPING: ContextVar[Dict[str, Dict[Device, LayerRepository]]] = ContextVar(
    "_KERNEL_MAPPING", default={}
)


def use_kernel_mapping(mapping: Dict[str, Dict[Device, LayerRepository]]):
    class ContextManager:
        def __enter__(self):
            # Mappings always stack on previous mappings.
            self.token = _KERNEL_MAPPING.set(deepcopy(_KERNEL_MAPPING.get()))
            register_kernel_mapping(mapping)

        def __exit__(self, exc_type, exc_value, traceback):
            _KERNEL_MAPPING.reset(self.token)

    return ContextManager()


def register_kernel_mapping(mapping: Dict[str, Dict[Device, LayerRepository]]):
    """
    Register a layer mapping.

    This function registers a mapping from a layer identifier and device type
    to a layer in a kernel repository.
    """
    # Merge with existing mappings.
    for new_kernel, new_device_repos in mapping.items():
        device_repo = _KERNEL_MAPPING.get().setdefault(new_kernel, {})
        for new_device, new_repo in new_device_repos.items():
            device_repo[new_device] = new_repo


def replace_kernel_forward_from_hub(cls, layer_name: str, *, use_fallback: bool = True):
    """
    Replace the forward function of a layer using a layer from the kernel hub.
    This function monkeypatches a layer, replacing the `forward` method
    of the layer with that of a layer from the hub. The replacement is done
    when a layer matching `layer_name` and device type is registered through
    `register_layer_mapping`. The device type is inferred from the first
    argument to `forward`.
    """

    fallback_forward = cls.forward

    cached_forward: Dict[LayerRepository, Callable] = {}

    def forward(self, x, **args):
        kernel = _KERNEL_MAPPING.get().get(layer_name)
        if kernel is None:
            if not use_fallback:
                raise ValueError(f"No layer mapping for `{layer_name}`")
            return fallback_forward(self, x, **args)

        device = getattr(x, "device", None)
        if device is None:
            return fallback_forward(self, x, **args)

        repo = kernel.get(Device(type=device.type))
        if repo is None:
            if not use_fallback:
                raise ValueError(
                    f"No layer mapping for `{layer_name}` with device type `{device.type}`"
                )
            return fallback_forward(self, x, **args)

        # Short-circuit if we already loaded the layer.
        layer_forward = cached_forward.get(repo, None)
        if layer_forward is not None:
            return layer_forward(self, x, **args)

        layer = _get_kernel_layer(
            repo_id=repo.repo_id,
            layer_name=repo.layer_name,
            revision=repo.revision,
        )

        # We have to validate against the original signature.
        orig_forward = cls.forward
        try:
            cls.forward = fallback_forward
            _validate_layer(check_cls=cls, cls=layer)
        finally:
            cls.forward = orig_forward

        layer_forward = layer.forward
        cached_forward[repo] = layer_forward

        return layer_forward(self, x, **args)

    cls.forward = forward


def use_kernel_forward_from_hub(layer_name: str, *, use_fallback: bool = True):
    """
    Replace the forward function of a layer using a layer from the kernel hub.
    This decorator can be applied to a layer and replaces the forward method
    of the layer with that of a layer from the hub. The replacement is done
    when a layer matching `layer_name` and device type is registered through
    `register_layer_mapping`. The device type is inferred from the first
    argument to `forward`.
    """

    def decorator(cls):
        replace_kernel_forward_from_hub(cls, layer_name, use_fallback=use_fallback)
        return cls

    return decorator


def _get_kernel_layer(*, repo_id: str, layer_name: str, revision: str) -> "nn.Module":
    """Get a layer from a kernel."""

    kernel = get_kernel(repo_id, revision=revision)

    if getattr(kernel, "layers", None) is None:
        raise ValueError(
            f"Kernel `{repo_id}` at revision `{revision}` does not define any layers."
        )

    layer = getattr(kernel.layers, layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{layer_name}` not found in kernel `{repo_id}`.")
    return layer


def _validate_layer(*, check_cls, cls):
    # The layer must have at least have the following properties: (1) it
    # must be stateless; (2) the forward signature should correspond to
    # the signature it is replacing; (3) forward should not call other
    # methods.

    from torch import nn

    if not issubclass(cls, nn.Module):
        raise TypeError(f"Layer `{cls}` is not a Torch layer.")

    # We verify statelessness by checking that the does not have its own
    # constructor (since the constructor could add member variables)...
    if cls.__init__ is not nn.Module.__init__:
        raise TypeError("Layer must not override nn.Module constructor.")

    # ... or predefined member variables.
    torch_module_members = {name for name, _ in inspect.getmembers(nn.Module)}
    cls_members = {name for name, _ in inspect.getmembers(cls)}
    if cls_members - torch_module_members != set():
        raise TypeError("Layer must not contain additional members.")

    # Check whether the forward signatures are similar.
    params = inspect.signature(cls.forward).parameters
    ref_params = inspect.signature(check_cls.forward).parameters

    if len(params) != len(ref_params):
        raise TypeError(
            "Forward signature does not match: different number of arguments."
        )

    for param, ref_param in zip(params.values(), ref_params.values()):
        if param.kind != ref_param.kind:
            raise TypeError(
                f"Forward signature does not match: different kind of arguments ({param} ({param.kind}) and {ref_param} ({ref_param.kind})"
            )
