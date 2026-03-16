#!/usr/bin/env python3
"""Detect the CUDA compute capability of the first available GPU device.

Exits with code 0 and prints the capability (e.g. "8.6") on success.
Exits with code 1 if the capability cannot be determined.
"""

import ctypes
import ctypes.util
import sys

# Definitions from cuda.h
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = ctypes.c_int(75)
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = ctypes.c_int(76)


def check_result(ret: int, msg: str) -> None:
    if ret != 0:
        print(f"{msg} (error code {ret})", file=sys.stderr)
        sys.exit(1)


def find_libcuda() -> ctypes.CDLL | None:
    candidates = [
        "libcuda.so.1",
        "libcuda.so",
        "/run/opengl-driver/lib/libcuda.so.1",
        "/run/opengl-driver/lib/libcuda.so",
        ctypes.util.find_library("cuda"),
    ]
    for name in candidates:
        if name is None:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def main() -> int:
    lib = find_libcuda()
    if lib is None:
        print("Could not load libcuda.so", file=sys.stderr)
        return 1

    # libcuda needs to be initialized before calling other functions.
    check_result(lib.cuInit(ctypes.c_uint(0)), "cuInit failed")

    # Get the first CUDA device.
    device = ctypes.c_int(0)
    check_result(
        lib.cuDeviceGet(ctypes.byref(device), ctypes.c_int(0)),
        "cuDeviceGet failed",
    )

    major = ctypes.c_int(0)
    check_result(
        lib.cuDeviceGetAttribute(
            ctypes.byref(major),
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        ),
        "cuDeviceGetAttribute (major) failed",
    )

    minor = ctypes.c_int(0)
    check_result(
        lib.cuDeviceGetAttribute(
            ctypes.byref(minor),
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        ),
        "cuDeviceGetAttribute (minor) failed",
    )

    print(f"{major.value}.{minor.value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
