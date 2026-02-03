function(cuda_kernel_component SRC_VAR)
    set(options SUPPORTS_HIPIFY)
    set(oneValueArgs CUDA_MINVER)
    set(multiValueArgs SOURCES INCLUDES CUDA_CAPABILITIES CUDA_FLAGS CXX_FLAGS HIP_FLAGS ROCM_ARCHS)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_SOURCES)
        message(FATAL_ERROR "cuda_kernel_component: SOURCES argument is required")
    endif()

    # Bail out if this component is not supported by the CUDA version.
    if(KERNEL_CUDA_MINVER)
        if(CUDA_VERSION VERSION_LESS ${KERNEL_CUDA_MINVER})
            return()
        endif()
    endif()

    set(_KERNEL_SRC ${KERNEL_SOURCES})

    if(KERNEL_INCLUDES)
        # TODO: check if CLion support this:
        # https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
        set_source_files_properties(
      ${_KERNEL_SRC}
      PROPERTIES INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}")
    endif()

    if(GPU_LANG STREQUAL "CUDA")
        # Determine CUDA architectures
        if(KERNEL_CUDA_CAPABILITIES)
            cuda_archs_loose_intersection(_KERNEL_ARCHS "${KERNEL_CUDA_CAPABILITIES}" "${CUDA_ARCHS}")
        else()
            set(_KERNEL_ARCHS "${CUDA_KERNEL_ARCHS}")
        endif()
        message(STATUS "CUDA kernel capabilities: ${_KERNEL_ARCHS}")
        set_gencode_flags_for_srcs(SRCS "${_KERNEL_SRC}" CUDA_ARCHS "${_KERNEL_ARCHS}")

        # Apply CUDA-specific compile flags
        if(KERNEL_CUDA_FLAGS)
            set(_CUDA_FLAGS "${KERNEL_CUDA_FLAGS}")
            # -static-global-template-stub is not supported on CUDA < 12.8. Remove this
            # once we don't support CUDA 12.6 anymore.
            if(CUDA_VERSION VERSION_LESS 12.8)
                string(REGEX REPLACE "-static-global-template-stub=(true|false)" "" _CUDA_FLAGS "${_CUDA_FLAGS}")
            endif()

            foreach(_SRC ${_KERNEL_SRC})
                if(_SRC MATCHES ".*\\.cu$")
                    set_property(
            SOURCE ${_SRC}
            APPEND PROPERTY
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_CUDA_FLAGS}>"
          )
                endif()
            endforeach()
        endif()

        # Apply CXX-specific compile flags
        if(KERNEL_CXX_FLAGS)
            foreach(_SRC ${_KERNEL_SRC})
                set_property(
          SOURCE ${_SRC}
          APPEND PROPERTY
          COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${KERNEL_CXX_FLAGS}>"
        )
            endforeach()
        endif()

        set(_TMP_SRC ${${SRC_VAR}})
        list(APPEND _TMP_SRC ${_KERNEL_SRC})
        set(${SRC_VAR} ${_TMP_SRC} PARENT_SCOPE)

    elseif(GPU_LANG STREQUAL "HIP")
        if(NOT KERNEL_SUPPORTS_HIPIFY)
            message(WARNING "Kernel does not support HIP")
            return()
        endif()

        # Apply HIP-specific compile flags
        if(KERNEL_HIP_FLAGS)
            foreach(_SRC ${_KERNEL_SRC})
                if(_SRC MATCHES ".*\\.(cu|hip)$")
                    set_property(
            SOURCE ${_SRC}
            APPEND PROPERTY
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:${KERNEL_HIP_FLAGS}>"
          )
                endif()
            endforeach()
        endif()

        # Determine ROCm architectures
        if(KERNEL_ROCM_ARCHS)
            hip_archs_loose_intersection(_KERNEL_ARCHS "${KERNEL_ROCM_ARCHS}" "${ROCM_ARCHS}")
        else()
            set(_KERNEL_ARCHS "${ROCM_ARCHS}")
        endif()
        message(STATUS "HIP kernel archs: ${_KERNEL_ARCHS}")

        foreach(_SRC ${_KERNEL_SRC})
            if(_SRC MATCHES ".*\\.(cu|hip)$")
                foreach(_ARCH ${_KERNEL_ARCHS})
                    set_property(
            SOURCE ${_SRC}
            APPEND PROPERTY
            COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${_ARCH}>"
          )
                endforeach()
            endif()
        endforeach()

        set(_TMP_SRC ${${SRC_VAR}})
        list(APPEND _TMP_SRC ${_KERNEL_SRC})
        set(${SRC_VAR} ${_TMP_SRC} PARENT_SCOPE)
    endif()
endfunction()

function(xpu_kernel_component SRC_VAR)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES INCLUDES CXX_FLAGS SYCL_FLAGS)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_SOURCES)
        message(FATAL_ERROR "xpu_kernel_component: SOURCES argument is required")
    endif()

    set(_KERNEL_SRC ${KERNEL_SOURCES})

    # Handle per-file include directories if specified
    if(KERNEL_INCLUDES)
        # TODO: check if CLion support this:
        # https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
        set_source_files_properties(
            ${_KERNEL_SRC}
            PROPERTIES INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}")
    endif()

    # Apply CXX-specific compile flags
    if(KERNEL_CXX_FLAGS)
        foreach(_SRC ${_KERNEL_SRC})
            set_property(
                SOURCE ${_SRC}
                APPEND PROPERTY
                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${KERNEL_CXX_FLAGS}>"
            )
        endforeach()
    endif()

    # Add SYCL-specific compilation flags for XPU sources
    if(KERNEL_SYCL_FLAGS)
        # Use kernel-specific SYCL flags
        foreach(_SRC ${_KERNEL_SRC})
            if(_SRC MATCHES ".*\\.(cpp|cxx|cc)$")
                set_property(
                    SOURCE ${_SRC}
                    APPEND PROPERTY
                    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${KERNEL_SYCL_FLAGS}>"
                )
            endif()
        endforeach()
    else()
        # Use default SYCL flags (from parent scope variable sycl_flags)
        foreach(_SRC ${_KERNEL_SRC})
            if(_SRC MATCHES ".*\\.(cpp|cxx|cc)$")
                set_property(
                    SOURCE ${_SRC}
                    APPEND PROPERTY
                    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${sycl_flags}>"
                )
            endif()
        endforeach()
    endif()

    # Append to parent scope SRC variable
    set(_TMP_SRC ${${SRC_VAR}})
    list(APPEND _TMP_SRC ${_KERNEL_SRC})
    set(${SRC_VAR} ${_TMP_SRC} PARENT_SCOPE)
endfunction()

function(cpu_kernel_component SRC_VAR)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES INCLUDES CXX_FLAGS)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_SOURCES)
        message(FATAL_ERROR "cpu_kernel_component: SOURCES argument is required")
    endif()

    set(_KERNEL_SRC ${KERNEL_SOURCES})

    # Handle per-file include directories if specified
    if(KERNEL_INCLUDES)
        # TODO: check if CLion support this:
        # https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
        set_source_files_properties(
            ${_KERNEL_SRC}
            PROPERTIES INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}")
    endif()

    # Apply CXX-specific compile flags
    if(KERNEL_CXX_FLAGS)
        foreach(_SRC ${_KERNEL_SRC})
            set_property(
                SOURCE ${_SRC}
                APPEND PROPERTY
                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${KERNEL_CXX_FLAGS}>"
            )
        endforeach()
    endif()

    # Append to parent scope SRC variable
    set(_TMP_SRC ${${SRC_VAR}})
    list(APPEND _TMP_SRC ${_KERNEL_SRC})
    set(${SRC_VAR} ${_TMP_SRC} PARENT_SCOPE)
endfunction()

function(metal_kernel_component SRC_VAR)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs SOURCES INCLUDES CXX_FLAGS)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_SOURCES)
        message(FATAL_ERROR "metal_kernel_component: SOURCES argument is required")
    endif()

    set(_KERNEL_SRC ${KERNEL_SOURCES})

    # Separate Metal shader files from other sources
    set(_METAL_SRC)
    set(_CPP_SRC)

    foreach(_SRC_FILE IN LISTS _KERNEL_SRC)
        if(_SRC_FILE MATCHES "\\.(metal|h)$")
            list(APPEND _METAL_SRC ${_SRC_FILE})
        else()
            list(APPEND _CPP_SRC ${_SRC_FILE})
        endif()
    endforeach()

    # Handle per-file include directories if specified (for C++ sources only)
    if(KERNEL_INCLUDES AND _CPP_SRC)
        # TODO: check if CLion support this:
        # https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
        set_source_files_properties(
            ${_CPP_SRC}
            PROPERTIES INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}")
    endif()

    # Apply CXX-specific compile flags
    if(KERNEL_CXX_FLAGS AND _CPP_SRC)
        foreach(_SRC ${_CPP_SRC})
            set_property(
                SOURCE ${_SRC}
                APPEND PROPERTY
                COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${KERNEL_CXX_FLAGS}>"
            )
        endforeach()
    endif()

    # Add C++ sources to main source list
    if(_CPP_SRC)
        set(_TMP_SRC ${${SRC_VAR}})
        list(APPEND _TMP_SRC ${_CPP_SRC})
        set(${SRC_VAR} ${_TMP_SRC} PARENT_SCOPE)
    endif()

    # Keep track of Metal sources for later compilation
    if(_METAL_SRC)
        set(_TMP_METAL ${ALL_METAL_SOURCES})
        list(APPEND _TMP_METAL ${_METAL_SRC})
        set(ALL_METAL_SOURCES ${_TMP_METAL} PARENT_SCOPE)
    endif()

    # Keep the includes directory for the Metal sources
    if(KERNEL_INCLUDES AND _METAL_SRC)
        set(_TMP_METAL_INCLUDES ${METAL_INCLUDE_DIRS})
        list(APPEND _TMP_METAL_INCLUDES ${KERNEL_INCLUDES})
        set(METAL_INCLUDE_DIRS ${_TMP_METAL_INCLUDES} PARENT_SCOPE)
    endif()
endfunction()
