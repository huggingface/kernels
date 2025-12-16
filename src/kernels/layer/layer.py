from __future__ import annotations

import functools
import inspect
import logging
import warnings
from pathlib import Path
from types import MethodType, ModuleType
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
    Protocol,
    Type,
)

from .device import Device
from .globals import _DISABLE_KERNEL_MAPPING, _KERNEL_MAPPING
from .._versions import select_revision_or_version
from ..utils import (
    _get_caller_locked_kernel,
    _get_locked_kernel,
    get_kernel,
    get_local_kernel,
)
from .mode import Mode
from .repos import _select_repository, RepositoryProtocol

if TYPE_CHECKING:
    from torch import nn


class LayerRepositoryProtocol(RepositoryProtocol, Protocol):
    @property
    def layer_name(self) -> str: ...


class LayerRepository:
    """
    Repository and name of a layer for kernel mapping.

    Args:
        repo_id (`str`):
            The Hub repository containing the layer.
        layer_name (`str`):
            The name of the layer within the kernel repository.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`str`, *optional*):
            The kernel version to download. This can be a Python version specifier, such as `">=1.0.0,<2.0.0"`.
            Cannot be used together with `revision`.

    Example:
        ```python
        from kernels import LayerRepository

        # Reference a specific layer by revision
        layer_repo = LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
        )

        # Reference a layer by version constraint
        layer_repo_versioned = LayerRepository(
            repo_id="kernels-community/activation",
            layer_name="SiluAndMul",
            version=">=0.0.3,<0.1"
        )
        ```
    """

    def __init__(
        self,
        repo_id: str,
        *,
        layer_name: str,
        revision: Optional[str] = None,
        version: Optional[str] = None,
    ):
        if revision is not None and version is not None:
            raise ValueError(
                "Either a revision or a version must be specified, not both."
            )

        self._repo_id = repo_id
        self.layer_name = layer_name

        # We are going to resolve these lazily, since we do not want
        # to do a network request for every registered LayerRepository.
        self._revision = revision
        self._version = version

    @functools.lru_cache()
    def _resolve_revision(self) -> str:
        return select_revision_or_version(
            repo_id=self._repo_id, revision=self._revision, version=self._version
        )

    def load(self) -> Type["nn.Module"]:
        kernel = get_kernel(self._repo_id, revision=self._resolve_revision())
        return _get_kernel_layer(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, LayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_id == other._repo_id
            and self._revision == other._revision
            and self._version == other._version
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_id, self._revision, self._version))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}), layer `{self.layer_name}`"


class LocalLayerRepository:
    """
    Repository from a local directory for kernel mapping.

    Args:
        repo_path (`Path`):
            The local repository containing the layer.
        package_name (`str`):
            Package name of the kernel.
        layer_name (`str`):
            The name of the layer within the kernel repository.

    Example:
        ```python
        from pathlib import Path

        from kernels import LocalLayerRepository

        # Reference a specific layer by revision
        layer_repo = LocalLayerRepository(
            repo_path=Path("/home/daniel/kernels/activation"),
            package_name="activation",
            layer_name="SiluAndMul",
        )
        ```
    """

    def __init__(
        self,
        repo_path: Path,
        *,
        package_name: str,
        layer_name: str,
    ):
        self._repo_path = repo_path
        self._package_name = package_name
        self.layer_name = layer_name

    def load(self) -> Type["nn.Module"]:
        kernel = get_local_kernel(self._repo_path, self._package_name)
        return _get_kernel_layer(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, LocalLayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_path == other._repo_path
            and self._package_name == other._package_name
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_path, self._package_name))

    def __str__(self) -> str:
        return f"`{self._repo_path}` (package: {self._package_name}), layer `{self.layer_name}`"


class LockedLayerRepository:
    """
    Repository and name of a layer.

    In contrast to `LayerRepository`, this class uses repositories that
    are locked inside a project.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        lockfile: Optional[Path] = None,
        layer_name: str,
    ):
        """
        Construct a layer repository.

        Args:
            repo_id (`str`): The Hub repository containing the layer.
        """
        self._repo_id = repo_id
        self._lockfile = lockfile
        self.layer_name = layer_name

    @functools.lru_cache()
    def _resolve_revision(self) -> str:
        if self._lockfile is None:
            locked_sha = _get_caller_locked_kernel(self._repo_id)
        else:
            with open(self._lockfile, "r") as f:
                locked_sha = _get_locked_kernel(self._repo_id, f.read())

        if locked_sha is None:
            raise ValueError(f"Kernel `{self._repo_id}` is not locked")

        return locked_sha

    def load(self) -> Type["nn.Module"]:
        kernel = get_kernel(repo_id=self._repo_id, revision=self._resolve_revision())
        return _get_kernel_layer(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, LockedLayerRepository)
            and self.layer_name == other.layer_name
            and self._repo_id == other._repo_id
        )

    def __hash__(self):
        return hash((self.layer_name, self._repo_id))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}), layer `{self.layer_name}`"


_CACHED_LAYER: Dict[RepositoryProtocol, Type["nn.Module"]] = {}


def replace_kernel_forward_from_hub(
    cls,
    layer_name: str,
):
    """
    Function that prepares a layer class to use kernels from the Hugging Face Hub.

    It is recommended to use [`use_kernel_forward_from_hub`] decorator instead.
    This function should only be used as a last resort to extend third-party layers,
    it is inherently fragile since the member variables and `forward` signature
    of such a layer can change.

    Example:
        ```python
        from kernels import replace_kernel_forward_from_hub
        import torch.nn as nn

        replace_kernel_forward_from_hub(nn.LayerNorm, "LayerNorm")
        ```
    """
    cls.kernel_layer_name = layer_name


def use_kernel_forward_from_hub(layer_name: str):
    """
    Decorator factory that makes a layer extensible using the specified layer name.

    This is a decorator factory that returns a decorator which prepares a layer class to use kernels from the
    Hugging Face Hub.

    Args:
        layer_name (`str`):
            The name of the layer to use for kernel lookup in registered mappings.

    Returns:
        `Callable`: A decorator function that can be applied to layer classes.

    Example:
        ```python
        import torch
        import torch.nn as nn

        from kernels import use_kernel_forward_from_hub
        from kernels import Mode, kernelize

        @use_kernel_forward_from_hub("MyCustomLayer")
        class MyCustomLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, x: torch.Tensor):
                # original implementation
                return x

        model = MyCustomLayer(768)

        # The layer can now be kernelized:
        # model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE, device="cuda")
        ```
    """

    def decorator(cls):
        replace_kernel_forward_from_hub(cls, layer_name)
        return cls

    return decorator


def kernelize_layer(
    module: "nn.Module", *, mode: Mode, device_type: Device, use_fallback
):
    module_class = type(module)
    layer_name = module_class.kernel_layer_name  # type: ignore[attr-defined]

    if _DISABLE_KERNEL_MAPPING:
        _replace_forward(module, module_class)
        return

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
        return

    # Get kernel options for the device
    property_repos = kernel.get(device_type.type)

    if property_repos is None:
        if not use_fallback:
            raise ValueError(
                f"No layer mapping for `{layer_name}` with device type `{device_type}`"
            )
        _replace_forward(module, module_class)
        return

    repos = property_repos.repos

    if repos is None:
        if not use_fallback:
            raise ValueError(
                f"No layer mapping for `{layer_name}` device `{device_type}` with the right properties"
            )
        _replace_forward(module, module_class)
        return

    repo_with_mode = _select_repository(
        repos,
        mode=mode,
    )

    if repo_with_mode is None:
        if not use_fallback:
            raise ValueError(
                f"No repository for `{layer_name}` for configuration mode={mode}"
            )
        _replace_forward(module, module_class)
        return

    repo, repo_mode = repo_with_mode

    logging.info(f"Using function/layer from repo {repo}")
    logging.debug(f"kernelize mode: {mode}, repo mode: {repo_mode}")

    layer = _get_layer_memoize(repo, module_class)

    # Ideally we would do validation on the mapping where we check that
    # e.g. if a repo class is registered for TRAINING | TORCH_COMPILE,
    # the actual layer is compatible with that. Unfortunately, this would
    # mean that we have to pre-download everything.
    _validate_layer_has_mode(
        layer_name=layer_name, module=layer, repo=repo, repo_mode=repo_mode
    )

    _conditionally_replace_forward(
        module=module,
        layer=layer,
        mode=mode,
        use_fallback=use_fallback,
    )


def _get_kernel_layer(
    repo: LayerRepositoryProtocol, kernel: ModuleType
) -> Type["nn.Module"]:
    """Get a layer from a kernel."""

    if getattr(kernel, "layers", None) is None:
        raise ValueError(f"Kernel repo {repo} does not define any layers.")

    layer = getattr(kernel.layers, repo.layer_name, None)
    if layer is None:
        raise ValueError(f"Layer `{repo.layer_name}` not found in kernel repo {repo}.")
    return layer


def _validate_layer(*, check_cls, cls, repo: RepositoryProtocol):
    import torch.nn as nn

    # The layer must have at least have the following properties: (1) it
    # must be stateless; (2) the forward signature should correspond to
    # the signature it is replacing; (3) forward should not call other
    # methods.

    if not issubclass(cls, nn.Module):
        raise TypeError(f"Layer `{cls.__name__}` is not a Torch layer.")

    # We verify statelessness by checking that the does not have its own
    # constructor (since the constructor could add member variables)...
    if cls.__init__ is not nn.Module.__init__:
        raise TypeError(f"{repo} must not override nn.Module constructor.")

    # ... or predefined member variables.
    torch_module_members = {name for name, _ in inspect.getmembers(nn.Module)}
    cls_members = {name for name, _ in inspect.getmembers(cls)}
    difference = cls_members - torch_module_members
    # verify if : difference ⊄ {"can_torch_compile", "has_backward"}
    if not difference <= {"can_torch_compile", "has_backward"}:
        raise TypeError(
            f"{repo} must not contain additional members compared to `{check_cls.__name__}`."
        )

    # Check whether the forward signatures are similar.
    params = inspect.signature(cls.forward).parameters
    ref_params = inspect.signature(check_cls.forward).parameters

    if len(params) != len(ref_params):
        raise TypeError(
            f"Forward signature of {repo} does not match `{check_cls.__name__}`: different number of arguments."
        )

    for param, ref_param in zip(params.values(), ref_params.values()):
        if param.kind != ref_param.kind:
            raise TypeError(
                f"Forward signature of {repo} does not match `{check_cls.__name__}`: different kind of arguments ({param} ({param.kind}) and {ref_param} ({ref_param.kind})"
            )


def _conditionally_replace_forward(
    *,
    module: "nn.Module",
    layer: Type["nn.Module"],
    mode: Mode,
    use_fallback: bool,
):
    module_class = type(module)

    # Switch to fallback if the mode is not supported by the layer.
    # Note that this is useful even after _validate_layer_has_mode because
    # layers registered with the FALLBACK mode never get rejected by
    # _validate_layer_has_mode. For such layers, we want to fall back in
    # case the layer does not support the given mode.
    needs_fallback_for_compile = Mode.TORCH_COMPILE in mode and not getattr(
        layer, "can_torch_compile", False
    )
    needs_fallback_for_backward = Mode.TRAINING in mode and not getattr(
        layer, "has_backward", True
    )

    if needs_fallback_for_compile or needs_fallback_for_backward:
        if use_fallback:
            if needs_fallback_for_compile:
                logging.info("Layer does not support torch.compile, using fallback")
            if needs_fallback_for_backward:
                logging.info("Layer does not support backward, using fallback")
            _replace_forward(module, module_class)
        else:
            raise ValueError(f"Available kernel does not support mode: {mode}")
    else:
        _replace_forward(module, layer)


def _replace_forward(module: "nn.Module", layer: Type["nn.Module"]):
    module.forward = MethodType(layer.forward, module)  # type: ignore[method-assign]


def _validate_layer_has_mode(
    *,
    layer_name: str,
    module: Type["nn.Module"],
    repo: RepositoryProtocol,
    repo_mode: Mode,
):
    """
    Check that a repository supports the mode that it was registered for.
    """

    if Mode.TRAINING in repo_mode and not getattr(module, "has_backward", True):
        raise ValueError(
            f"Function/layer from repo {repo} does not support backward.\n"
            f"Was registered for `{layer_name}` with mode `{repo_mode}`"
        )

    if Mode.TORCH_COMPILE in repo_mode and not getattr(
        module, "can_torch_compile", False
    ):
        raise ValueError(
            f"Function/layer from repo {repo} does not support torch.compile.\n"
            f"Was registered for `{layer_name}` with mode `{repo_mode}`"
        )

    return True


def _get_layer_memoize(
    repo: RepositoryProtocol, module_class: Type["nn.Module"]
) -> Type["nn.Module"]:
    layer = _CACHED_LAYER.get(repo, None)
    if layer is not None:
        return layer

    layer = repo.load()
    _validate_layer(check_cls=module_class, cls=layer, repo=repo)
    _CACHED_LAYER[repo] = layer

    return layer
