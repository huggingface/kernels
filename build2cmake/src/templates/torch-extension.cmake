# Include Metal shader compilation utilities if needed
if(GPU_LANG STREQUAL "METAL")
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/compile-metal.cmake)
endif()

# Define the extension target with unified parameters
define_gpu_extension_target(
  {{ ops_name }}
  DESTINATION {{ ops_name }}
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${GPU_FLAGS}
  ARCHITECTURES ${GPU_ARCHES}
  USE_SABI 3
  WITH_SOABI)

if(NOT (MSVC OR GPU_LANG STREQUAL "SYCL"))
  target_link_options({{ ops_name }} PRIVATE -static-libstdc++)
endif()

if(GPU_LANG STREQUAL "SYCL")
  target_link_options({{ ops_name }} PRIVATE ${sycl_link_flags})
  target_link_libraries({{ ops_name }} PRIVATE dnnl)
endif()

# Compile Metal shaders if any were found
if(GPU_LANG STREQUAL "METAL")
  if(ALL_METAL_SOURCES)
    compile_metal_shaders({{ ops_name }} "${ALL_METAL_SOURCES}" "${METAL_INCLUDE_DIRS}")
  endif()
endif()

# Add kernels_install target for huggingface/kernels library layout
add_kernels_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")

# Add local_install target for local development with get_local_kernel()
add_local_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")
