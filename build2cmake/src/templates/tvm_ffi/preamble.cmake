cmake_minimum_required(VERSION 3.26)

project({{name}} LANGUAGES CXX)

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)

# Replace by detection logic.
set(BACKEND "cuda" CACHE STRING "Backend to build for")
set(GPU_LANG "CUDA" CACHE STRING "GPU language")

find_package(Python COMPONENTS Interpreter REQUIRED)

set(KERNEL_REVISION "{{ revision }}" CACHE STRING "Kernel revision, must be unique")
set(OPS_NAME "_{{python_name}}_${BACKEND}_{{ revision }}")

option(BUILD_ALL_SUPPORTED_ARCHS "Build all supported architectures" on)

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0+PTX")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX")
else()
  set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX")
endif()

# Make conditional on the toolkit found in the environment.
enable_language(CUDA)

if(GPU_LANG STREQUAL "CUDA")
  # Get the capabilities without +PTX suffixes, so that we can use them as
  # the target archs in the loose intersection with a kernel's capabilities.
  cuda_remove_ptx_suffixes(CUDA_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  message(STATUS "CUDA supported base architectures: ${CUDA_ARCHS}")

  if(BUILD_ALL_SUPPORTED_ARCHS)
    set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  else()
    # TODO: detect capability.
    message(FATAL_ERROR "Capability detection is not implemented for CUDA yet, please set BUILD_ALL_SUPPORTED_ARCHS to ON to build for all supported architectures.")
  endif()
endif()

# Run `tvm-ffi-config --cmakedir` to set `tvm_ffi_ROOT`
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
find_package(tvm_ffi CONFIG REQUIRED)

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/_ops.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/tvm-ffi-ext/{{python_name}}/_ops.py
  @ONLY
)
