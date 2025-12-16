import functools
import inspect
from inspect import Parameter, Signature
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Type

from kernels.layer.repos import RepositoryProtocol

from .._versions import select_revision_or_version
from ..utils import (
    _get_caller_locked_kernel,
    _get_locked_kernel,
    get_kernel,
    get_local_kernel,
)

if TYPE_CHECKING:
    from torch import nn


class FuncRepositoryProtocol(RepositoryProtocol, Protocol):
    @property
    def func_name(self) -> str: ...


class FuncRepository:
    """
    Repository and name of a function for kernel mapping.

    Args:
        repo_id (`str`):
            The Hub repository containing the layer.
        func_name (`str`):
            The name of the function within the kernel repository.
        revision (`str`, *optional*, defaults to `"main"`):
            The specific revision (branch, tag, or commit) to download. Cannot be used together with `version`.
        version (`str`, *optional*):
            The kernel version to download. This can be a Python version specifier, such as `">=1.0.0,<2.0.0"`.
            Cannot be used together with `revision`.

    Example:
        ```python
        from kernels import FuncRepository

        # Reference a specific layer by revision
        layer_repo = FuncRepository(
            repo_id="kernels-community/activation",
            func_name="silu_and_mul",
        )

        # Reference a layer by version constraint
        layer_repo_versioned = FuncRepository(
            repo_id="kernels-community/activation",
            func_name="silu_and_mul",
            version=">=0.0.3,<0.1"
        )
        ```
    """

    def __init__(
        self,
        repo_id: str,
        *,
        func_name: str,
        revision: Optional[str] = None,
        version: Optional[str] = None,
    ):
        if revision is not None and version is not None:
            raise ValueError(
                "Either a revision or a version must be specified, not both."
            )

        self._repo_id = repo_id
        self.func_name = func_name

        # We are going to resolve these lazily, since we do not want
        # to do a network request for every registered FuncRepository.
        self._revision = revision
        self._version = version

    @functools.lru_cache()
    def _resolve_revision(self) -> str:
        return select_revision_or_version(
            repo_id=self._repo_id, revision=self._revision, version=self._version
        )

    def load(self) -> Type["nn.Module"]:
        kernel = get_kernel(self._repo_id, revision=self._resolve_revision())
        return _get_kernel_func(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, FuncRepository)
            and self.func_name == other.func_name
            and self._repo_id == other._repo_id
            and self._revision == other._revision
            and self._version == other._version
        )

    def __hash__(self):
        return hash((self.func_name, self._repo_id, self._revision, self._version))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}), function `{self.func_name}`"


class LocalFuncRepository:
    """
    Repository and function name from a local directory for kernel mapping.

    Args:
        repo_path (`Path`):
            The local repository containing the layer.
        package_name (`str`):
            Package name of the kernel.
        func_name (`str`):
            The name of the function within the kernel repository.

    Example:
        ```python
        from pathlib import Path

        from kernels import LocalFuncRepository

        # Reference a specific layer by revision
        layer_repo = LocalFuncRepository(
            repo_path=Path("/home/daniel/kernels/activation"),
            package_name="activation",
            func_name="silu_and_mul",
        )
        ```
    """

    def __init__(
        self,
        repo_path: Path,
        *,
        package_name: str,
        func_name: str,
    ):
        self._repo_path = repo_path
        self._package_name = package_name
        self.func_name = func_name

    def load(self) -> Type["nn.Module"]:
        kernel = get_local_kernel(self._repo_path, self._package_name)
        return _get_kernel_func(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, LocalFuncRepository)
            and self.func_name == other.func_name
            and self._repo_path == other._repo_path
            and self._package_name == other._package_name
        )

    def __hash__(self):
        return hash((self.func_name, self._repo_path, self._package_name))

    def __str__(self) -> str:
        return f"`{self._repo_path}` (package: {self._package_name}), layer `{self.func_name}`"


def use_kernel_func_from_hub(func_name: str):
    """
    Decorator that makes a function extensible using the specified function name.

    This is a decorator factory that returns a decorator which prepares a function to use kernels from the
    Hugging Face Hub.

    The function will be exposed as an instance of `torch.nn.Module` in which
    the function is called in `forward`. For the function to be properly
    kernelized, it **must** be a member of another `torch.nn.Module` that is
    part of the model (see the example).

    Args:
        func_name (`str`):
            The name of the function name to use for kernel lookup in registered mappings.

    Returns:
        `Callable`: A decorator function that can be applied to layer classes.

    Example:
        ```python
        import torch
        import torch.nn as nn

        from kernels import use_kernel_func_from_hub
        from kernels import Mode, kernelize

        @use_kernel_func_from_hub("my_custom_func")
        def my_custom_func(x: torch.Tensor):
            # Original implementation
            return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fn = my_custom_func

            def forward(self, x):
                return self.fn(x)

        model = MyModel()

        # The layer can now be kernelized:
        # model = kernelize(model, mode=Mode.TRAINING | Mode.TORCH_COMPILE, device="cuda")
        ```
    """

    def decorator(func):
        Func = _create_func_module(func)
        Func.kernel_layer_name = func_name
        return Func()

    return decorator


class LockedFuncRepository:
    """
    Repository and name of a function.

    In contrast to `FuncRepository`, this class uses repositories that
    are locked inside a project.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        lockfile: Optional[Path] = None,
        func_name: str,
    ):
        """
        Construct a function repository.

        Args:
            repo_id (`str`): The Hub repository containing the function.
            lockfile (`Path`, *optional*): Path to the lockfile. If not provided,
                the lockfile will be inferred from the caller's context.
            func_name (`str`): The name of the function within the kernel repository.
        """
        self._repo_id = repo_id
        self._lockfile = lockfile
        self.func_name = func_name

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
        return _get_kernel_func(self, kernel)

    def __eq__(self, other):
        return (
            isinstance(other, LockedFuncRepository)
            and self.func_name == other.func_name
            and self._repo_id == other._repo_id
        )

    def __hash__(self):
        return hash((self.func_name, self._repo_id))

    def __str__(self) -> str:
        return f"`{self._repo_id}` (revision: {self._resolve_revision()}), function `{self.func_name}`"


def _get_kernel_func(
    repo: FuncRepositoryProtocol, kernel: ModuleType
) -> Type["nn.Module"]:
    func = getattr(kernel, repo.func_name, None)
    if func is None:
        raise ValueError(f"Function `{repo.func_name}` not found in `{repo}`")

    return _create_func_module(func)


def _create_func_module(func: Callable) -> Type["nn.Module"]:
    from torch import nn

    class Func(nn.Module):
        def forward(self, *args, **kwargs):
            return func(*args, **kwargs)

    # Use function signature with args prepended by self to support
    # module validation.
    func_sig = inspect.signature(func)
    new_args = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
    new_args.extend(func_sig.parameters.values())
    Func.forward.__signature__ = Signature(  # type: ignore[attr-defined]
        parameters=new_args,
        return_annotation=func_sig.return_annotation,
    )

    return Func
