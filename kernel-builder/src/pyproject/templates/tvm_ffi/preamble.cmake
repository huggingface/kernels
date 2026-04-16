cmake_minimum_required(VERSION 3.26)

# Set Intel SYCL compiler before project() call
find_program(ICX_COMPILER icx)
find_program(ICPX_COMPILER icpx)

if(ICX_COMPILER OR ICPX_COMPILER)
  set(CMAKE_C_COMPILER ${ICX_COMPILER})

 if(WIN32)
    set(CMAKE_CXX_COMPILER ${ICX_COMPILER})
  else()
    set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
  endif()
endif()

project({{name}} LANGUAGES CXX)

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)

if(NOT DEFINED GPU_LANG)
    if(ICX_COMPILER OR ICPX_COMPILER)
        set(DETECTED_GPU_LANG "SYCL")
    else()
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            set(DETECTED_GPU_LANG "CUDA")
        else()
            set(DETECTED_GPU_LANG "CPU")
        endif()
    endif()

    set(GPU_LANG "${DETECTED_GPU_LANG}" CACHE STRING "GPU language")
endif()
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

set(OPS_NAME "_{{kernel_name}}_${BACKEND}_{{ kernel_unique_id }}")

option(BUILD_ALL_SUPPORTED_ARCHS "Build all supported architectures" off)

if(GPU_LANG STREQUAL "CUDA")
  enable_language(CUDA)

  {% if cuda_minver %}
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS {{ cuda_minver }})
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} is too old. "
      "Minimum required version is {{ cuda_minver }}.")
  endif()
  {% endif %}

  {% if cuda_maxver %}
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER {{ cuda_maxver }})
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} is too new. "
      "Maximum version is {{ cuda_maxver }}.")
  endif()
  {% endif %}

  if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0+PTX")
  elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
      CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX")
  else()
    set(CUDA_DEFAULT_KERNEL_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX")
  endif()

  # We have per-source file archs, so disable global arch setting.
  set(CMAKE_CUDA_ARCHITECTURES OFF)

  # Get the capabilities without +PTX suffixes, so that we can use them as
  # the target archs in the loose intersection with a kernel's capabilities.
  cuda_remove_ptx_suffixes(CUDA_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  message(STATUS "CUDA base archs used for intersection with kernel archs: ${CUDA_ARCHS}")

  if(BUILD_ALL_SUPPORTED_ARCHS)
      set(CUDA_KERNEL_ARCHS "${CUDA_DEFAULT_KERNEL_ARCHS}")
  else()
      # Detect the compute capability of the first available GPU device.
      run_python_script(DETECTED_CUDA_CAPABILITY
          "${CMAKE_CURRENT_LIST_DIR}/cmake/cuda/detect-cuda-capability.py"
          "Cannot detect CUDA device capability. Set BUILD_ALL_SUPPORTED_ARCHS=ON to disable detection.")
      message(STATUS "Detected CUDA device capability: ${DETECTED_CUDA_CAPABILITY}")
      set(CUDA_KERNEL_ARCHS "${DETECTED_CUDA_CAPABILITY}")
  endif()

  message(STATUS "Default CUDA kernel architectures: ${CUDA_KERNEL_ARCHS}")

  if(NVCC_THREADS AND GPU_LANG STREQUAL "CUDA")
    message(STATUS "Using nvcc with: -threads=${NVCC_THREADS}")
    list(APPEND GPU_FLAGS "--threads=${NVCC_THREADS}")
  endif()

  add_compile_definitions(CUDA_KERNEL)
elseif(GPU_LANG STREQUAL "CPU")
  add_compile_definitions(CPU_KERNEL)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING "Minimum macOS deployment version")
elseif(GPU_LANG STREQUAL "SYCL")
  if(NOT ICX_COMPILER AND NOT ICPX_COMPILER)
    message(FATAL_ERROR "Intel SYCL C++ compiler (icpx) and/or C compiler (icx) not found. Please install Intel oneAPI toolkit.")
  endif()

  execute_process(
    COMMAND ${ICPX_COMPILER} --version
    OUTPUT_VARIABLE ICPX_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "[0-9]+\\.[0-9]+" DPCPP_VERSION "${ICPX_VERSION_OUTPUT}")
  set(DPCPP_VERSION "${DPCPP_VERSION}" CACHE STRING "DPCPP major.minor version")

  # On Windows, use icx (MSVC-compatible) for C++ to work with Ninja generator
  # On Linux, use icpx (GNU-compatible) for C++
  if(WIN32)
    message(STATUS "Using Intel SYCL C++ compiler: ${ICX_COMPILER} and C compiler: ${ICX_COMPILER} Version: ${DPCPP_VERSION} (Windows MSVC-compatible mode)")
  else()
    message(STATUS "Using Intel SYCL C++ compiler: ${ICPX_COMPILER} and C compiler: ${ICX_COMPILER} Version: ${DPCPP_VERSION}")
  endif()

  set(sycl_link_flags "-fsycl;--offload-compress;-fsycl-targets=spir64_gen,spir64;-Xs;-device pvc,xe-lpg,ats-m150 -options ' -cl-intel-enable-auto-large-GRF-mode -cl-poison-unsupported-fp64-kernels -cl-intel-greater-than-4GB-buffer-required';")
  set(sycl_flags "-fsycl;-fhonor-nans;-fhonor-infinities;-fno-associative-math;-fno-approx-func;-fno-sycl-instrument-device-code;--offload-compress;-fsycl-targets=spir64_gen,spir64;")
  set(GPU_FLAGS "${sycl_flags}")
  set(GPU_ARCHES "")

  add_compile_definitions(XPU_KERNEL)
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
