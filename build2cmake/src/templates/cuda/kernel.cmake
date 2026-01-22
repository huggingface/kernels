{% if cuda_minver %}
if (CUDA_VERSION VERSION_GREATER_EQUAL {{ cuda_minver }})
{% endif %}

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

if(GPU_LANG STREQUAL "CUDA")
  {% if cuda_capabilities %}
    cuda_archs_loose_intersection({{kernel_name}}_ARCHS "{{ cuda_capabilities|join(";") }}" "${CUDA_ARCHS}")
  {% else %}
    set({{kernel_name}}_ARCHS "${CUDA_KERNEL_ARCHS}")
  {% endif %}
  message(STATUS "Capabilities for kernel {{kernel_name}}: {{ '${' + kernel_name + '_ARCHS}'}}")
  set_gencode_flags_for_srcs(SRCS {{'"${' + kernel_name + '_SRC}"'}} CUDA_ARCHS "{{ '${' + kernel_name + '_ARCHS}'}}")

  {% if cuda_flags %}

  set(_CUDA_FLAGS "{{ cuda_flags }}")
  # -static-global-template-stub is not supported on CUDA < 12.8. Remove this
  # once we don't support CUDA 12.6 anymore.
  if(CUDA_VERSION VERSION_LESS 12.8)
    string(REGEX REPLACE "-static-global-template-stub=(true|false)" "" _CUDA_FLAGS "${_CUDA_FLAGS}")
  endif()

  foreach(_KERNEL_SRC {{'${' + kernel_name + '_SRC}'}})
    if(_KERNEL_SRC MATCHES ".*\\.cu$")
      set_property(
        SOURCE ${_KERNEL_SRC}
        APPEND PROPERTY
        COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_CUDA_FLAGS}>"
      )
    endif()
  endforeach()
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

  list(APPEND SRC {{'"${' + kernel_name + '_SRC}"'}})
{% if supports_hipify %}
elseif(GPU_LANG STREQUAL "HIP")
  {% if hip_flags %}

  foreach(_KERNEL_SRC {{'${' + kernel_name + '_SRC}'}})
    if(_KERNEL_SRC MATCHES ".*\\.(cu|hip)$")
      set_property(
        SOURCE ${_KERNEL_SRC}
        APPEND PROPERTY
        COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:{{ hip_flags }}>"
      )
    endif()
  endforeach()
  {% endif %}

  hip_archs_loose_intersection({{kernel_name}}_ARCHS "{{ rocm_archs|join(";") }}" "${ROCM_ARCHS}")
  message(STATUS "Archs for kernel {{kernel_name}}: {{ '${' + kernel_name + '_ARCHS}'}}")

  foreach(_KERNEL_SRC {{'${' + kernel_name + '_SRC}'}})
    if(_KERNEL_SRC MATCHES ".*\\.(cu|hip)$")
      foreach(_ROCM_ARCH {{ '${' + kernel_name + '_ARCHS}'}})
        set_property(
          SOURCE ${_KERNEL_SRC}
          APPEND PROPERTY
          COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${_ROCM_ARCH}>"
        )
      endforeach()
    endif()
  endforeach()

  list(APPEND SRC {{'"${' + kernel_name + '_SRC}"'}})
{% endif %}
endif()

{% if cuda_minver %}
endif()
{% endif %}
