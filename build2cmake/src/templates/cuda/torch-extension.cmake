define_gpu_extension_target(
  {{ ops_name }}
  DESTINATION {{ ops_name }}
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${GPU_FLAGS}
  ARCHITECTURES ${GPU_ARCHES}
  #INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if( NOT MSVC)
    target_link_options({{ ops_name }} PRIVATE -static-libstdc++)
endif()

{% if platform == 'windows' %}
# These methods below should be included from preamble.cmake on windows platform.

# Add kernels_install target for huggingface/kernels library layout
add_kernels_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")

# Add local_install target for local development with get_local_kernel()
add_local_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")

{% endif %}
