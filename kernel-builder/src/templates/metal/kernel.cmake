if(GPU_LANG STREQUAL "METAL")
  metal_kernel_component(SRC
    SOURCES {{ sources }}
    {% if includes %}INCLUDES "{{ includes }}"{% endif %}
    {% if cxx_flags %}CXX_FLAGS "{{ cxx_flags }}"{% endif %}
  )
endif()
