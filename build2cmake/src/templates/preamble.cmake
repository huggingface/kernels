cmake_minimum_required(VERSION 3.26)
project({{name}} LANGUAGES CXX)

set(TARGET_DEVICE "cuda" CACHE STRING "Target device backend for kernel")

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1200;gfx1201")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/get_gpu_lang.cmake)

if(DEFINED Python3_EXECUTABLE)
  # Allow passing through the interpreter (e.g. from setup.py).
  find_package(Python3 COMPONENTS Development Development.SABIModule Interpreter)
  if (NOT Python3_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
else()
  find_package(Python3 REQUIRED COMPONENTS Development Development.SABIModule Interpreter)
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

find_package(Torch REQUIRED)

run_python(TORCH_VERSION "import torch; print(torch.__version__.split('+')[0])" "Failed to get Torch version")

get_gpu_lang(DETECTED_GPU_LANG)
set(GPU_LANG "${DETECTED_GPU_LANG}" CACHE STRING "GPU language")
message(STATUS "Using GPU language: ${GPU_LANG}")

{% if torch_minver %}
if (TORCH_VERSION VERSION_LESS {{ torch_minver }})
  message(FATAL_ERROR "Torch version ${TORCH_VERSION} is too old. "
    "Minimum required version is {{ torch_minver }}.")
endif()
{% endif %}

{% if torch_maxver %}
if (TORCH_VERSION VERSION_GREATER {{ torch_maxver }})
  message(FATAL_ERROR "Torch version ${TORCH_VERSION} is too new. "
    "Maximum supported version is {{ torch_maxver }}.")
endif()
{% endif %}

if (NOT TARGET_DEVICE STREQUAL "cuda" AND
    NOT TARGET_DEVICE STREQUAL "rocm")
    return()
endif()

option(BUILD_ALL_SUPPORTED_ARCHS "Build all supported architectures" off)

if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0+PTX")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
 set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX")
else()
  set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX")
endif()


# Basic checks for each GPU language.
if(GPU_LANG STREQUAL "CUDA")
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "GPU language is set to CUDA, but cannot find CUDA toolkit")
  endif()

  {% if cuda_minver %}
    if (CUDA_VERSION VERSION_LESS {{ cuda_minver }})
      message(FATAL_ERROR "CUDA version ${CUDA_VERSION} is too old. "
        "Minimum required version is {{ cuda_minver }}.")
    endif()
  {% endif %}

  {% if cuda_maxver %}
    if (CUDA_VERSION VERSION_GREATER {{ cuda_maxver }})
      message(FATAL_ERROR "CUDA version ${CUDA_VERSION} is too new. "
        "Maximum version is {{ cuda_maxver }}.")
    endif()
  {% endif %}

  # TODO: deprecate one of these settings.
  add_compile_definitions(USE_CUDA=1)
  add_compile_definitions(CUDA_KERNEL)
elseif(GPU_LANG STREQUAL "HIP")
  if(NOT HIP_FOUND)
    message(FATAL_ERROR "GPU language is set to HIP, but cannot find ROCm toolkit")
  endif()

  # Importing torch recognizes and sets up some HIP/ROCm configuration but does
  # not let cmake recognize .hip files. In order to get cmake to understand the
  # .hip extension automatically, HIP must be enabled explicitly.
  enable_language(HIP)

  # TODO: deprecate one of these settings.
  add_compile_definitions(USE_ROCM=1)
  add_compile_definitions(ROCM_KERNEL)
endif()

# CUDA build options.
if(GPU_LANG STREQUAL "CUDA")
  # This clears out -gencode arguments from `CMAKE_CUDA_FLAGS`, which we need
  # to set our own set of capabilities.
  clear_gencode_flags()

  # Get the capabilities without +PTX suffixes, so that we can use them as
  # the target archs in the loose intersection with a kernel's capabilities.
  cuda_remove_ptx_suffixes(CUDA_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  message(STATUS "CUDA supported base architectures: ${CUDA_ARCHS}")

  if(BUILD_ALL_SUPPORTED_ARCHS)
    set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  else()
    try_run_python(CUDA_KERNEL_ARCHS SUCCESS "import torch; cc=torch.cuda.get_device_capability(); print(f\"{cc[0]}.{cc[1]}\")" "Failed to get CUDA capability")
    if(NOT SUCCESS)
      message(WARNING "Failed to detect CUDA capability, using default capabilities.")
      set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
    endif()
  endif()

  message(STATUS "CUDA supported kernel architectures: ${CUDA_KERNEL_ARCHS}")

  if(NVCC_THREADS AND GPU_LANG STREQUAL "CUDA")
    list(APPEND GPU_FLAGS "--threads=${NVCC_THREADS}")
  endif()

elseif(GPU_LANG STREQUAL "HIP")
  override_gpu_arches(GPU_ARCHES HIP ${HIP_SUPPORTED_ARCHS})
  set(ROCM_ARCHS ${GPU_ARCHES})
  message(STATUS "ROCM supported target architectures: ${ROCM_ARCHS}")

else()
  override_gpu_arches(GPU_ARCHES
    ${GPU_LANG}
    "${${GPU_LANG}_SUPPORTED_ARCHS}")
endif()

# Initialize SRC list for kernel and binding sources
set(SRC "")

message(STATUS "Rendered for platform {{ platform }}")

{% if platform == 'windows' %}
include(${CMAKE_CURRENT_LIST_DIR}/cmake/windows.cmake)

# Generate standardized build name
cmake_host_system_information(RESULT HOST_ARCH QUERY OS_PLATFORM)

set(SYSTEM_STRING "${HOST_ARCH}-windows")

if(GPU_LANG STREQUAL "CUDA")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "cuda" "${CUDA_VERSION}" "${SYSTEM_STRING}")
elseif(GPU_LANG STREQUAL "HIP")
  run_python(ROCM_VERSION "import torch.version; print(torch.version.hip.split('.')[0] + '.' + torch.version.hip.split('.')[1])" "Failed to get ROCm version")
  generate_build_name(BUILD_VARIANT_NAME "${TORCH_VERSION}" "rocm" "${ROCM_VERSION}" "${SYSTEM_STRING}")
endif()
{% endif %}
