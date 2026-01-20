set({{kernel_name}}_SRC
  {{ sources }}
)

# Separate Metal shader files from other sources
set({{kernel_name}}_METAL_SRC)
set({{kernel_name}}_CPP_SRC)

foreach(src_file IN LISTS {{kernel_name}}_SRC)
  if(src_file MATCHES "\\.(metal|h)$")
    list(APPEND {{kernel_name}}_METAL_SRC ${src_file})
  else()
    list(APPEND {{kernel_name}}_CPP_SRC ${src_file})
  endif()
endforeach()

{% if includes %}
# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  {{'${' + kernel_name + '_CPP_SRC}'}}
  PROPERTIES INCLUDE_DIRECTORIES "{{ includes }}")
{% endif %}

{% if cxx_flags %}
foreach(_KERNEL_SRC {{'${' + kernel_name + '_CPP_SRC}'}})
  set_property(
    SOURCE ${_KERNEL_SRC}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:{{ cxx_flags }}>"
  )
endforeach()
{% endif %}

# Add C++ sources to main source list
list(APPEND SRC {{'"${' + kernel_name + '_CPP_SRC}"'}})

# Keep track of Metal sources for later compilation
if({{kernel_name}}_METAL_SRC)
  list(APPEND ALL_METAL_SOURCES {{'"${' + kernel_name + '_METAL_SRC}"'}})
endif()

{% if includes %}
# Keep the includes directory for the Metal sources
if({{kernel_name}}_METAL_SRC)
  list(APPEND METAL_INCLUDE_DIRS {{ includes }})
endif()
{% endif %}