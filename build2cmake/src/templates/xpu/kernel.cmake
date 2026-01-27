xpu_kernel_component(SRC
  SOURCES {{ sources }}
  {% if includes %}INCLUDES "{{ includes }}"{% endif %}
  {% if cxx_flags %}CXX_FLAGS "{{ cxx_flags }}"{% endif %}
  {% if sycl_flags %}SYCL_FLAGS "{{ sycl_flags }}"{% endif %}
)
