set(TVM_FFI_{{name}}_SRC
  {{ src|join(' ') }}
)

{% if includes %}
# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  {{'${TVM_FFI_' + name + '_SRC}'}}
  PROPERTIES INCLUDE_DIRECTORIES "{{ includes }}")
{% endif %}

list(APPEND SRC {{'"${TVM_FFI_' + name + '_SRC}"'}})
