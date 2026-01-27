cuda_kernel_component(SRC
  SOURCES {{ sources }}
  {% if cuda_minver %}CUDA_MINVER {{ cuda_minver }}{% endif %}
  {% if includes %}INCLUDES "{{ includes }}"{% endif %}
  {% if cuda_capabilities %}CUDA_CAPABILITIES {{ cuda_capabilities|join(" ") }}{% endif %}
  {% if cuda_flags %}CUDA_FLAGS "{{ cuda_flags }}"{% endif %}
  {% if cxx_flags %}CXX_FLAGS "{{ cxx_flags }}"{% endif %}
  {% if supports_hipify %}SUPPORTS_HIPIFY{% endif %}
  {% if hip_flags %}HIP_FLAGS "{{ hip_flags }}"{% endif %}
  {% if rocm_archs %}ROCM_ARCHS {{ rocm_archs|join(" ") }}{% endif %}
)
