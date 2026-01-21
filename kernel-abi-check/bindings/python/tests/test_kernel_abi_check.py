from pathlib import Path

import pytest

from kernel_abi_check import (
    BinaryFormat,
    IncompatibleAbi3Symbol,
    IncompatibleMacOSVersion,
    ObjectFile,
    IncompatibleManylinuxSymbol,
)


@pytest.fixture
def test_dir():
    return Path(__file__).parent


def test_macos_shared_lib(test_dir):
    o = ObjectFile(test_dir / "hello-darwin-x86_64.abi3.so")
    assert o.check_python_abi("3.8") == []
    assert o.check_python_abi("3.5") == [
        IncompatibleAbi3Symbol(name="PyModule_GetNameObject", added="3.7")
    ]

    assert o.check_macos("15.0") == []
    assert o.check_macos("10.0") == [IncompatibleMacOSVersion(version="11.3")]

    assert o.format() == BinaryFormat.MACH_O


def test_linux_shared_lib(test_dir):
    o = ObjectFile(test_dir / "hello-linux-x86_64.abi3.so")
    assert o.check_python_abi("3.8") == []
    assert o.check_python_abi("3.5") == [
        IncompatibleAbi3Symbol(name="PyModule_GetNameObject", added="3.7")
    ]

    assert o.check_manylinux("manylinux_2_34") == []
    assert o.check_manylinux("manylinux_2_28") == [
        IncompatibleManylinuxSymbol(name="fstat64", dep="GLIBC", version="2.33"),
        IncompatibleManylinuxSymbol(
            name="pthread_key_create", dep="GLIBC", version="2.34"
        ),
        IncompatibleManylinuxSymbol(
            name="pthread_key_delete", dep="GLIBC", version="2.34"
        ),
        IncompatibleManylinuxSymbol(
            name="pthread_setspecific", dep="GLIBC", version="2.34"
        ),
        IncompatibleManylinuxSymbol(name="stat64", dep="GLIBC", version="2.33"),
    ]

    assert o.format() == BinaryFormat.ELF
