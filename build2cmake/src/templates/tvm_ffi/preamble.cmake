cmake_minimum_required(VERSION 3.26)

project({{name}} LANGUAGES CXX)

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(DETECTED_GPU_LANG "CUDA")
else()
    set(DETECTED_GPU_LANG "CPU")
endif()

set(GPU_LANG "${DETECTED_GPU_LANG}" CACHE STRING "GPU language")
gpu_lang_to_backend(BACKEND "${GPU_LANG}")
message(STATUS "Using backend: ${BACKEND}, GPU language: ${GPU_LANG}")

if(DEFINED Python_EXECUTABLE)
  # Allow passing through the interpreter (e.g. from setup.py).
  find_package(Python COMPONENTS Development Development.SABIModule Interpreter)
  if (NOT Python_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
else()
  find_package(Python REQUIRED COMPONENTS Development Development.SABIModule Interpreter)
endif()

set(KERNEL_REVISION "{{ revision }}" CACHE STRING "Kernel revision, must be unique")
set(OPS_NAME "_{{python_name}}_${BACKEND}_{{ revision }}")

option(BUILD_ALL_SUPPORTED_ARCHS "Build all supported architectures" on)

if(GPU_LANG STREQUAL "CUDA")
  enable_language(CUDA)

  if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0+PTX")
  elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX")
  else()
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX")
  endif()

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

  add_compile_definitions(CUDA_KERNEL)
elseif(GPU_LANG STREQUAL "CPU")
  add_compile_definitions(CPU_KERNEL)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment version")
endif()

# Run `tvm-ffi-config --cmakedir` to set `tvm_ffi_ROOT`
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
find_package(tvm_ffi CONFIG REQUIRED)

run_python(TVM_FFI_VERSION "import tvm_ffi; print(tvm_ffi.__version__.split('-')[0])" "Failed to get tvm-ffi version")
message(STATUS "Found tvm-ffi version: ${TVM_FFI_VERSION}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/build-variants.cmake)

# Generate build variant name.
if(GPU_LANG STREQUAL "CUDA")
  generate_build_name(BUILD_VARIANT_NAME "${TVM_FFI_VERSION}" "cuda" "${CMAKE_CUDA_COMPILER_VERSION}")
elseif(GPU_LANG STREQUAL "HIP")
  generate_build_name(BUILD_VARIANT_NAME "${TVM_FFI_VERSION}" "rocm" "${ROCM_VERSION}")
elseif(GPU_LANG STREQUAL "SYCL")
  generate_build_name(BUILD_VARIANT_NAME "${TVM_FFI_VERSION}" "xpu" "${DPCPP_VERSION}")
elseif(GPU_LANG STREQUAL "METAL")
  generate_build_name(BUILD_VARIANT_NAME "${TVM_FFI_VERSION}" "metal" "")
elseif(GPU_LANG STREQUAL "CPU")
  generate_build_name(BUILD_VARIANT_NAME "${TVM_FFI_VERSION}" "cpu" "")
else()
  message(FATAL_ERROR "Cannot generate build name for unknown GPU_LANG: ${GPU_LANG}")
endif()

configure_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/_ops.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/tvm-ffi-ext/{{python_name}}/_ops.py
  @ONLY
)
