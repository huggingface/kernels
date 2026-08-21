if(GPU_LANG STREQUAL "CPU")
rust_kernel_component(RUST_KERNEL_LIBS RUST_KERNEL_TARGETS
  NAME {{ name }}
  MANIFEST_PATH "{{ manifest_path }}"
  LIB_NAME {{ lib_name }}
  {% if features %}FEATURES {{ features|join(" ") }}{% endif %}
)
endif()
