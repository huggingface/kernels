define_gpu_extension_target(
  {{ ops_name }}
  DESTINATION {{ ops_name }}
  LANGUAGE ${GPU_LANG}
  SOURCES ${SRC}
  COMPILE_FLAGS ${sycl_flags}
  USE_SABI 3
  WITH_SOABI)

# Add XPU/SYCL specific linker flags
target_link_options({{ ops_name }} PRIVATE ${sycl_link_flags})
target_link_libraries({{ ops_name }} PRIVATE dnnl)

{% if platform == 'windows' %}
# These methods below should be included from preamble.cmake on windows platform.

# Add kernels_install target for huggingface/kernels library layout
add_kernels_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")

# Add local_install target for local development with get_local_kernel()
add_local_install_target({{ ops_name }} "{{ name }}" "${BUILD_VARIANT_NAME}")

{% endif %}