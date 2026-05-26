if(GPU_LANG STREQUAL "CUDA")
cuda_kernel_component(SRC
  NAME {{ name }}
  SOURCES {{ sources }}
  {% if cuda_minver %}CUDA_MINVER {{ cuda_minver }}{% endif %}
  {% if includes %}INCLUDES "{{ includes }}"{% endif %}
  {% if cuda_capabilities %}CUDA_CAPABILITIES {{ cuda_capabilities|join(" ") }}{% endif %}
  {% if cuda_flags %}CUDA_FLAGS "{{ cuda_flags }}"{% endif %}
  {% if cxx_flags %}CXX_FLAGS "{{ cxx_flags }}"{% endif %}
)
endif()
