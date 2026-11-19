from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .repos import RepositoryProtocol

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class KernelizedLayer:
    """
    A kernelized layer and the repository that its kernel was retrieved from.

    Attributes:
        module (`nn.Module`):
            The kernelized layer.
        repo (`RepositoryProtocol`):
            The repository that the kernel was retrieved from.
    """

    module: nn.Module
    repo: RepositoryProtocol


@dataclass(frozen=True)
class KernelizedFunc:
    """
    A kernelized function and the repository that its kernel was retrieved from.

    Attributes:
        module (`nn.Module`):
            The module that wraps the kernelized function.
        repo (`RepositoryProtocol`):
            The repository that the kernel was retrieved from.
    """

    module: nn.Module
    repo: RepositoryProtocol


@dataclass(frozen=True)
class KernelizedLayers:
    """
    The kernels that are in use by a kernelized model, as returned by
    [`~kernels.get_kernelized_layers`].

    Attributes:
        layers (`dict[str, KernelizedLayer]`):
            A mapping of each kernelized layer name to the layer and its repository. The
            names are the same as returned by Torch's `named_modules`.
        funcs (`dict[tuple[str, str], KernelizedFunc]`):
            A mapping of each kernelized function to the function and its repository. The
            key of the mapping is a tuple of the layer name that the function is attached
            to and the name of the function itself.
    """

    layers: dict[str, KernelizedLayer] = field(default_factory=dict)
    funcs: dict[tuple[str, str], KernelizedFunc] = field(default_factory=dict)


def get_kernelized_layers(model: "nn.Module") -> KernelizedLayers:
    """
    Return the repositories of the kernel layers and functions that are used by the model.

    Args:
        model (`nn.Module`):
            The kernelized model.

    Returns:
        [`~kernels.KernelizedLayers`]: The kernelized layers.

    Example:
        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        from kernels import LayerRepository, Mode, get_kernelized_layers, kernelize
        from kernels import use_kernel_forward_from_hub, use_kernel_mapping

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
                    version=1,
                )
            }
        }

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_proj = nn.Linear(1024, 2048)
                self.act_fn = SiluAndMul()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.act_fn(self.up_proj(x))

        model = MLP().to("cuda")

        with use_kernel_mapping(mapping):
            model = kernelize(model, mode=Mode.INFERENCE)

        kernelized = get_kernelized_layers(model)
        print(kernelized.layers)
        # {'act_fn': KernelizedLayer(module=..., repo=LayerRepository(...))}
        print(kernelized.funcs)
        # {} (no kernel functions are attached to this model)
        ```
    """
    kernelized = KernelizedLayers()

    for module_name, module in model.named_modules():
        repo = module.__dict__.get("_kernel_repo")
        if repo is not None:
            kernelized.layers[module_name] = KernelizedLayer(module, repo)

        # Kernel functions attached through `use_kernelized_func` are plain
        # attributes, so they are not part of `named_modules`.
        for func_name, kernel_func in getattr(module, "_kernel_funcs", {}).items():
            repo = kernel_func.__dict__.get("_kernel_repo")
            if repo is not None:
                kernelized.funcs[(module_name, func_name)] = KernelizedFunc(kernel_func, repo)

    return kernelized
