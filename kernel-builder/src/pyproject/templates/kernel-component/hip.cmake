if(GPU_LANG STREQUAL "HIP")
hip_kernel_component(SRC
  NAME {{ name }}
  SOURCES {{ sources }}
  {% if includes %}INCLUDES "{{ includes }}"{% endif %}
  {% if cxx_flags %}CXX_FLAGS "{{ cxx_flags }}"{% endif %}
  {% if hip_flags %}HIP_FLAGS "{{ hip_flags }}"{% endif %}
  {% if rocm_archs %}ROCM_ARCHS {{ rocm_archs|join(" ") }}{% endif %}
)
endif()
