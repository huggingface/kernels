set(TORCH_{{name}}_SRC
  {{ src|join(' ') }}
)

{% if includes %}
# TODO: check if CLion support this:
# https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
set_source_files_properties(
  {{'${TORCH_' + name + '_SRC}'}}
  PROPERTIES INCLUDE_DIRECTORIES "{{ includes }}")
{% endif %}

list(APPEND SRC {{'"${TORCH_' + name + '_SRC}"'}})
