set({{kernel_name}}_SRC
  {{ sources }}
)

{% if includes %}
# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  {{'${' + kernel_name + '_SRC}'}}
  PROPERTIES INCLUDE_DIRECTORIES "{{ includes }}")
{% endif %}

{% if cxx_flags %}
foreach(_KERNEL_SRC {{'${' + kernel_name + '_SRC}'}})
  set_property(
    SOURCE ${_KERNEL_SRC}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:{{ cxx_flags }}>"
  )
endforeach()
{% endif %}

# Add C++ sources to main source list
list(APPEND SRC {{'"${' + kernel_name + '_SRC}"'}})
