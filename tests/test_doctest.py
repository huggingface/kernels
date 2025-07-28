import inspect

import pytest
from mktestdocs import check_docstring, get_codeblock_members

import kernels


def all_public_functions():
    function_list = inspect.getmembers(kernels, inspect.isfunction)
    return [func for _, func in function_list]


def all_public_classes():
    class_list = inspect.getmembers(kernels, inspect.isclass)
    return [cls for _, cls in class_list]


def all_public_class_members():
    members = get_codeblock_members(*all_public_classes())
    return members


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "func",
    all_public_functions(),
    ids=lambda d: d.__name__,
)
def test_func_docstring(func):
    check_docstring(obj=func)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "cls",
    all_public_classes(),
    ids=lambda d: d.__name__,
)
def test_class_docstring(cls):
    check_docstring(obj=cls)


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "member", all_public_class_members(), ids=lambda d: d.__qualname__
)
def test_member_docstring(member):
    check_docstring(member)
