from __future__ import annotations

from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Union,
)

from .repos import DeviceRepos
from .globals import _KERNEL_MAPPING
from .layer import kernelize_layer
from .repos import RepositoryProtocol
from .mode import Mode
from .device import Device

if TYPE_CHECKING:
    import torch
    from torch import nn


def use_kernel_mapping(
    mapping: Dict[
        str,
        Dict[
            Union[Device, str],
            Union[RepositoryProtocol, Dict[Mode, RepositoryProtocol]],
        ],
    ],
    *,
    inherit_mapping: bool = True,
):
    """
    Context manager that sets a kernel mapping for the duration of the context.

    This function allows temporary kernel mappings to be applied within a specific context, enabling different
    kernel configurations for different parts of your code.

    Args:
        mapping (`Dict[str, Dict[Union[Device, str], Union[LayerRepositoryProtocol, Dict[Mode, LayerRepositoryProtocol]]]]`):
            The kernel mapping to apply. Maps layer names to device-specific kernel configurations.
        inherit_mapping (`bool`, *optional*, defaults to `True`):
            When `True`, the current mapping will be extended by `mapping` inside the context. When `False`,
            only `mapping` is used inside the context.

    Returns:
        Context manager that handles the temporary kernel mapping.

    Example:
        ```python
        import torch
        import torch.nn as nn
        from torch.nn import functional as F

        from kernels import use_kernel_forward_from_hub
        from kernels import use_kernel_mapping, LayerRepository, Device
        from kernels import Mode, kernelize

        # Define a mapping
        mapping = {
            "SiluAndMul": {
                "cuda": LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="SiluAndMul",
                )
            }
        }

        @use_kernel_forward_from_hub("SiluAndMul")
        class SiluAndMul(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return F.silu(x[..., :d]) * x[..., d:]

        model = SiluAndMul()

        # Use the mapping for the duration of the context.
        with use_kernel_mapping(mapping):
            # kernelize uses the temporary mapping
            model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE, device="cuda")

        # Outside the context, original mappings are restored
        ```
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
    mapping: Dict[
        str,
        Dict[
            Union[Device, str],
            Union[RepositoryProtocol, Dict[Mode, RepositoryProtocol]],
        ],
    ],
    inherit_mapping: bool = True,
):
    """
    Register a global mapping between layer names and their corresponding kernel implementations.

    This function allows you to register a mapping between a layer name and the corresponding kernel(s) to use,
    depending on the device and mode. This should be used in conjunction with [`kernelize`].

    Args:
        mapping (`Dict[str, Dict[Union[Device, str], Union[RepositoryProtocol, Dict[Mode, RepositoryProtocol]]]]`):
            The kernel mapping to register globally. Maps layer names to device-specific kernels.
            The mapping can specify different kernels for different modes (training, inference, etc.).
        inherit_mapping (`bool`, *optional*, defaults to `True`):
            When `True`, the current mapping will be extended by `mapping`. When `False`, the existing mappings
            are erased before adding `mapping`.

    Example:
        ```python
        from kernels import LayerRepository, register_kernel_mapping, Mode

        # Simple mapping for a single kernel per device
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

        # Advanced mapping with mode-specific kernels
        advanced_mapping = {
            "MultiHeadAttention": {
                "cuda": {
                    Mode.TRAINING: LayerRepository(
                        repo_id="username/training-kernels",
                        layer_name="TrainingAttention"
                    ),
                    Mode.INFERENCE: LayerRepository(
                        repo_id="username/inference-kernels",
                        layer_name="FastAttention"
                    ),
                }
            }
        }
        register_kernel_mapping(advanced_mapping)
        ```
    """
    if not inherit_mapping:
        _KERNEL_MAPPING.set({})

    # Merge with existing mappings.
    for new_kernel, new_device_repos in mapping.items():
        device_repo = _KERNEL_MAPPING.get().setdefault(new_kernel, {})
        for new_device, new_repo in new_device_repos.items():
            device = (
                Device(type=new_device) if isinstance(new_device, str) else new_device
            )

            if isinstance(new_repo, dict):
                kernel_options = new_repo
            else:
                kernel_options = {Mode.FALLBACK: new_repo}

            feature_repos = device_repo.setdefault(
                device.type, DeviceRepos.create_repo(device)
            )
            feature_repos.insert(device, kernel_options)


def kernelize(
    model: "nn.Module",
    *,
    mode: Mode,
    device: Optional[Union[str, "torch.device"]] = None,
    use_fallback: bool = True,
):
    """
    Replace layer forward methods with optimized kernel implementations.

    This function iterates over all modules in the model and replaces the `forward` method of extensible layers
    for which kernels are registered using [`register_kernel_mapping`] or [`use_kernel_mapping`].

    Args:
        model (`nn.Module`):
            The PyTorch model to kernelize.
        mode ([`Mode`]): The mode that the kernel is going to be used in. For example,
            `Mode.TRAINING | Mode.TORCH_COMPILE` kernelizes the model for training with
            `torch.compile`.
        device (`Union[str, torch.device]`, *optional*):
            The device type to load kernels for. Supported device types are: "cuda", "mps", "npu", "rocm", "xpu".
            The device type will be inferred from the model parameters when not provided.
        use_fallback (`bool`, *optional*, defaults to `True`):
            Whether to use the original forward method of modules when no compatible kernel could be found.
            If set to `False`, an exception will be raised in such cases.

    Returns:
        `nn.Module`: The kernelized model with optimized kernel implementations.

    Example:
        ```python
        import torch
        import torch.nn as nn

        from kernels import kernelize, Mode, register_kernel_mapping, LayerRepository
        from kernels import use_kernel_forward_from_hub

        @use_kernel_forward_from_hub("SiluAndMul")
        class SiluAndMul(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                d = x.shape[-1] // 2
                return F.silu(x[..., :d]) * x[..., d:]

        mapping = {
            "SiluAndMul": {
                "cuda": LayerRepository(
                    repo_id="kernels-community/activation",
                    layer_name="SiluAndMul",
                )
            }
        }
        register_kernel_mapping(mapping)

        # Create and kernelize a model
        model = nn.Sequential(
            nn.Linear(1024, 2048, device="cuda"),
            SiluAndMul(),
        )

        # Kernelize for inference
        kernelized_model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE)
        ```
    """

    if mode == Mode.FALLBACK:
        raise ValueError("Mode.FALLBACK can only be used to register kernel mappings.")

    # Type check ignored because this causes a false negative on Python < 3.11.
    # Looks similar to: https://github.com/python/mypy/issues/9642
    # Remove once we start doing typing checks on >= 3.11.
    if Mode.INFERENCE not in mode and Mode.TRAINING not in mode:  # type: ignore[operator]
        raise ValueError("kernelize mode must contain Mode.INFERENCE or Mode.TRAINING.")

    if device is None:
        device_type = _find_device(model)
    elif isinstance(device, str):
        _validate_device_type(device)
        device_type = Device(type=device)
    else:
        device_type = Device(device.type)

    assert isinstance(device_type, Device)

    for _, module in model.named_modules():
        module_class = type(module)
        if not hasattr(module_class, "kernel_layer_name"):
            continue

        kernelize_layer(
            module, mode=mode, device_type=device_type, use_fallback=use_fallback
        )

    return model


def _validate_device_type(device_type: str) -> None:
    """Validate that the device type is supported."""
    supported_devices = {"cpu", "cuda", "mps", "npu", "rocm", "xpu"}
    if device_type not in supported_devices:
        raise ValueError(
            f"Unsupported device type '{device_type}'. Supported device types are: {', '.join(sorted(supported_devices))}"
        )


def _find_device(model: "nn.Module") -> Device:
    try:
        param = next(model.parameters())
    except StopIteration:
        raise ValueError(
            "Cannot determine model device, provide as `device` argument to `kernelize`."
        )

    dev_type = param.device.type
    if dev_type == "cuda":
        # Refine based on actual platform
        if _is_rocm_platform():
            return Device(type="rocm")
        elif _is_cuda_platform():
            return Device(type="cuda")

    return Device(type=dev_type)


def _is_cuda_platform():
    import torch

    return torch.version.cuda is not None


def _is_rocm_platform():
    import torch

    return torch.version.hip is not None
