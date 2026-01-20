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
