set(TVM_FFI_{{name}}_SRC
  {{ src|join(' ') }}
)

# only append binding sources if the source list is non-empty
if(TVM_FFI_{{name}}_SRC)
{% if includes %}
# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  {{'${TVM_FFI_' + name + '_SRC}'}}
  PROPERTIES INCLUDE_DIRECTORIES "{{ includes }}")
{% endif %}

{% if cxx_flags %}
set_property(
  SOURCE {{'${TVM_FFI_' + name + '_SRC}'}}
  APPEND PROPERTY
  COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:{{ cxx_flags }}>")
{% endif %}

list(APPEND SRC {{'${TVM_FFI_' + name + '_SRC}'}})
endif()
