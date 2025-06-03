import inspect
import os
import warnings
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Dict, Optional, Type, Union

from .utils import get_kernel

if TYPE_CHECKING:
    import torch
    from torch import nn


_DISABLE_KERNEL_MAPPING: bool = bool(int(os.environ.get("DISABLE_KERNEL_MAPPING", "0")))


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


_CACHED_LAYER: Dict[LayerRepository, Type["nn.Module"]] = {}


_KERNEL_MAPPING: ContextVar[Dict[str, Dict[Device, LayerRepository]]] = ContextVar(
    "_KERNEL_MAPPING", default={}
)


def use_kernel_mapping(
    mapping: Dict[str, Dict[Union[Device, str], LayerRepository]],
    *,
    inherit_mapping: bool = True,
):
    """
    Context manager that sets a mapping for a duration of the context.

    When `inherit_mapping` is set to `True` the current mapping will be
    extended by `mapping` inside the context. If it is `False`, only
    `mapping` is used inside the context.
    """

    class ContextManager:
        def __enter__(self):
            # Mappings always stack on previous mappings.
            if inherit_mapping:
                self.token = _KERNEL_MAPPING.set(deepcopy(_KERNEL_MAPPING.get()))
            else:
                self.token = _KERNEL_MAPPING.set({})
            register_kernel_mapping(mapping)

        def __exit__(self, exc_type, exc_value, traceback):
            _KERNEL_MAPPING.reset(self.token)

    return ContextManager()


def register_kernel_mapping(
    mapping: Dict[str, Dict[Union[Device, str], LayerRepository]],
):
    """
    Allows one to register a mapping between a layer name the corresponding
    kernel to use, depending on the device. This should be use in conjunction
    with `kernelize`.

    Exemple usage:

    ```python
    from kernels import LayerRepository, register_kernel_mapping

    kernel_layer_mapping = {
      "LlamaRMSNorm": {
          "cuda": LayerRepository(
              repo_id="kernels-community/activation",
              layer_name="RmsNorm",
              revision="layers",
          ),
      },
    }
    register_kernel_mapping(kernel_layer_mapping)
    ```
    """
    # Merge with existing mappings.
    for new_kernel, new_device_repos in mapping.items():
        device_repo = _KERNEL_MAPPING.get().setdefault(new_kernel, {})
        for new_device, new_repo in new_device_repos.items():
            if isinstance(new_device, str):
                device_repo[Device(type=new_device)] = new_repo
            else:
                device_repo[new_device] = new_repo


def replace_kernel_forward_from_hub(
    cls,
    layer_name: str,
):
    """
    Decorator that prepares a layer class to use a kernel from the Hugging Face Hub.

    This decorator stores the layer name and original forward method, which will be used
    by the kernelize function to replace the forward implementation with the appropriate
    kernel from the hub.

    Args:
        cls: The layer class to decorate
        layer_name: The name of the layer to use for kernel lookup
    """
    cls.kernel_layer_name = layer_name


def kernelize(
    model: "nn.Module",
    device: Optional[Union[str, "torch.device"]] = None,
    needs_torch_compile: bool = False,
    use_fallback: bool = True,
):
    """
    Iterate over all modules in the model and replace the `forward` method of
    extensible layers for which kernels are registered using `register_kernel_mapping`
    or `use_kernel_mapping`.

    Args:
        model: The PyTorch model to kernelize
        device: The device type to load kernels for. The device type will be inferred
            from the parameters of the model when not provided.
        needs_torch_compile: When set to `true`, only kernels that support
            `torch.compile` will be loaded.
        use_fallback: Whether to use the original forward method of modules when no
            compatible kernel could be found. If set to `False`, an exception will
            be raised in such cases.

    Returns:
        The kernelized model
    """
    import torch

    if device is None:
        device_type = _find_device(model)
    elif isinstance(device, str):
        device_type = Device(type=torch.device(device).type)
    else:
        device_type = Device(device.type)
    assert isinstance(device_type, Device)

    for _, module in model.named_modules():
        module_class = type(module)
        if not hasattr(module_class, "kernel_layer_name"):
            continue
        layer_name = module_class.kernel_layer_name

        if _DISABLE_KERNEL_MAPPING:
            _replace_forward(module, module_class)
            continue

        kernel = _KERNEL_MAPPING.get().get(str(layer_name))

        if kernel is None:
            warnings.warn(
                "\n"
                f"No kernel mapping found for layer `{layer_name}`. "
                f"Check if the layer name matches one of the kernels in the mapping or add the kernel "
                f"you want to use to the mapping. Defaulting to original forward implementation."
            )
            if not use_fallback:
                raise ValueError(f"No layer mapping for `{layer_name}`")
            _replace_forward(module, module_class)
            continue

        # Use device type string directly instead of Device object
        repo = kernel.get(device_type)

        if repo is None:
            if not use_fallback:
                raise ValueError(
                    f"No layer mapping for `{layer_name}` with device type `{device_type}`"
                )
            _replace_forward(module, module_class)
            continue

        # Short-circuit if we already loaded the layer.
        layer = _CACHED_LAYER.get(repo, None)
        if layer is not None:
            _conditionally_replace_forward(
                module=module,
                layer=layer,
                needs_torch_compile=needs_torch_compile,
                use_fallback=use_fallback,
            )
            continue

        layer = _get_kernel_layer(
            repo_id=repo.repo_id,
            layer_name=repo.layer_name,
            revision=repo.revision,
        )

        # Validate the replacement layer against the class layer.
        _validate_layer(check_cls=module_class, cls=layer)

        _CACHED_LAYER[repo] = layer

        _conditionally_replace_forward(
            module=module,
            layer=layer,
            needs_torch_compile=needs_torch_compile,
            use_fallback=use_fallback,
        )

    return model


def use_kernel_forward_from_hub(layer_name: str):
    """
    Make a layer extensible using the name `layer_name`.
    """

    def decorator(cls):
        replace_kernel_forward_from_hub(cls, layer_name)
        return cls

    return decorator


def _get_kernel_layer(
    *, repo_id: str, layer_name: str, revision: str
) -> Type["nn.Module"]:
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
    import torch.nn as nn

    # The layer must have at least have the following properties: (1) it
    # must be stateless; (2) the forward signature should correspond to
    # the signature it is replacing; (3) forward should not call other
    # methods.

    if not issubclass(cls, nn.Module):
        raise TypeError(f"Layer `{cls}` is not a Torch layer.")

    # We verify statelessness by checking that the does not have its own
    # constructor (since the constructor could add member variables)...
    if cls.__init__ is not nn.Module.__init__:
        raise TypeError("Layer must not override nn.Module constructor.")

    # ... or predefined member variables.
    torch_module_members = {name for name, _ in inspect.getmembers(nn.Module)}
    cls_members = {name for name, _ in inspect.getmembers(cls)}
    difference = cls_members - torch_module_members
    # verify if : difference ⊄ {"can_torch_compile", "has_backward"}
    if not difference <= {"can_torch_compile", "has_backward"}:
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


def _find_device(model: "nn.Module") -> Device:
    try:
        param = next(model.parameters())
    except StopIteration:
        raise ValueError(
            "Cannot determine model device, provide as `device` argument to `kernelize`."
        )

    return Device(type=param.device.type)


def _conditionally_replace_forward(
    *,
    module: "nn.Module",
    layer: Type["nn.Module"],
    needs_torch_compile: bool,
    use_fallback: bool,
):
    module_class = type(module)

    # Switch to fallback when the layer does not support:
    # compilation/compile when needed.
    # backward when needed
    needs_fallback = needs_torch_compile and not getattr(
        layer, "can_torch_compile", False
    )
    if needs_fallback:
        if use_fallback:
            _replace_forward(module, module_class)
        else:
            raise ValueError(
                f"Available kernel does not fulfill requirements: needs_torch_compile={needs_torch_compile}"
            )
    else:
        _replace_forward(module, layer)


def _replace_forward(module: "nn.Module", layer: Type["nn.Module"]):
    import torch.nn as nn

    module_class = type(module)
    layer_with_backward = (
        layer if getattr(layer, "has_backward", True) else module_class
    )

    def train(self, mode: bool = True) -> nn.Module:
        super(type(self), self).train(mode)
        if mode:
            self.forward = MethodType(layer_with_backward.forward, self)
        else:
            self.forward = MethodType(layer.forward, self)
        return self

    module.train = MethodType(train, module)  # type: ignore[method-assign]

    # Trigger setting correct forward for the current state.
    module.train(module.training)
