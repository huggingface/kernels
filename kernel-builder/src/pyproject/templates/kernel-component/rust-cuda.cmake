if(GPU_LANG STREQUAL "CUDA")
rust_kernel_component(RUST_KERNEL_LIBS RUST_KERNEL_TARGETS
  NAME {{ name }}
  MANIFEST_PATH "{{ manifest_path }}"
  LIB_NAME {{ lib_name }}
  {% if features %}FEATURES {{ features|join(" ") }}{% endif %}
  {% if cuda_capabilities %}CUDA_CAPABILITIES {{ cuda_capabilities|join(" ") }}{% endif %}
)
endif()
