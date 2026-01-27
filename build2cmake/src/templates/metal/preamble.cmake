cmake_minimum_required(VERSION 3.26)
project({{name}} LANGUAGES CXX C OBJC OBJCXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS deployment version")

install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR}) # Ensure the directory exists
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/kernel.cmake)

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

add_compile_definitions(METAL_KERNEL)

# Initialize SRC list for kernel and binding sources
set(SRC "")

# Initialize lists for Metal shader sources and their include directories
set(ALL_METAL_SOURCES)
set(METAL_INCLUDE_DIRS)
