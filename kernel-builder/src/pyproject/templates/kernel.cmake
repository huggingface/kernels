function(accumulate_gpu_archs OUT_ACC ACC EXTRA_ARCHS)
    list(APPEND ACC ${EXTRA_ARCHS})
    list(REMOVE_DUPLICATES ACC)
    list(SORT ACC)
    set(${OUT_ACC} ${ACC} PARENT_SCOPE)
endfunction()

function(cuda_kernel_component SRC_VAR)
    set(oneValueArgs CUDA_MINVER NAME)
    set(multiValueArgs SOURCES INCLUDES CUDA_CAPABILITIES CUDA_FLAGS CXX_FLAGS)
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

    # Determine CUDA architectures
    if(KERNEL_CUDA_CAPABILITIES)
        cuda_archs_loose_intersection(_KERNEL_ARCHS "${KERNEL_CUDA_CAPABILITIES}" "${CUDA_ARCHS}")
        if(NOT _KERNEL_ARCHS)
            message(FATAL_ERROR "CUDA kernel: ${KERNEL_NAME}, empty set of capabilities after intersection (kernel: ${KERNEL_CUDA_CAPABILITIES}, supported: ${CUDA_ARCHS})")
        endif()
    else()
        set(_KERNEL_ARCHS "${CUDA_KERNEL_ARCHS}")
    endif()
    message(STATUS "CUDA kernel: ${KERNEL_NAME}, capabilities: ${_KERNEL_ARCHS}")
    set_gencode_flags_for_srcs(SRCS "${_KERNEL_SRC}" CUDA_ARCHS "${_KERNEL_ARCHS}")

    accumulate_gpu_archs(_ALL_GPU_ARCHS "${ALL_GPU_ARCHS}" "${_KERNEL_ARCHS}")
    set(ALL_GPU_ARCHS ${_ALL_GPU_ARCHS} PARENT_SCOPE)

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
endfunction()

function(hip_kernel_component SRC_VAR)
    set(options SUPPORTS_HIPIFY)
    set(oneValueArgs CUDA_MINVER NAME)
    set(multiValueArgs SOURCES INCLUDES CXX_FLAGS HIP_FLAGS ROCM_ARCHS)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_SOURCES)
        message(FATAL_ERROR "hip_kernel_component: SOURCES argument is required")
    endif()

    set(_KERNEL_SRC ${KERNEL_SOURCES})

    if(KERNEL_INCLUDES)
        # TODO: check if CLion support this:
        # https://youtrack.jetbrains.com/issue/CPP-16510/CLion-does-not-handle-per-file-include-directories
        set_source_files_properties(
      ${_KERNEL_SRC}
      PROPERTIES INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}")
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
        if(NOT _KERNEL_ARCHS)
            message(FATAL_ERROR "ROCm kernel: ${KERNEL_NAME}, empty set of architectures after intersection (kernel: ${KERNEL_ROCM_ARCHS}, supported: ${ROCM_ARCHS})")
        endif()
    else()
        set(_KERNEL_ARCHS "${ROCM_ARCHS}")
    endif()
    message(STATUS "ROCm kernel: ${KERNEL_NAME}, archs: ${_KERNEL_ARCHS}")

    accumulate_gpu_archs(_ALL_GPU_ARCHS "${ALL_GPU_ARCHS}" "${_KERNEL_ARCHS}")
    set(ALL_GPU_ARCHS ${_ALL_GPU_ARCHS} PARENT_SCOPE)

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
        # Use kernel-specific SYCL flags plus SYCL flags from the parent scope.
        set(_SYCL_FLAGS ${sycl_flags})
        list(APPEND _SYCL_FLAGS ${KERNEL_SYCL_FLAGS})
        foreach(_SRC ${_KERNEL_SRC})
            if(_SRC MATCHES ".*\\.(cpp|cxx|cc)$")
                set_property(
                    SOURCE ${_SRC}
                    APPEND PROPERTY
                    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${_SYCL_FLAGS}>"
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

# Compile TIRx modules (Python) to CUDA C++ sources. Returns the generated
# sources in SRC_VAR; the caller passes them to cuda_kernel_component so that
# they get the same architecture and flag handling as hand-written CUDA.
function(tirx_codegen SRC_VAR)
    cmake_parse_arguments(TIRX "" "NAME" "SRC;CUDA_CAPABILITIES" ${ARGN})

    if(NOT TIRX_NAME)
        message(FATAL_ERROR "tirx_codegen requires NAME")
    endif()

    # Codegen targets a single architecture; nvcc then compiles the generated
    # source for every requested architecture. Use the oldest one so that the
    # generated source does not rely on intrinsics newer targets provide.
    if(TIRX_CUDA_CAPABILITIES)
        cuda_archs_loose_intersection(_tirx_archs "${TIRX_CUDA_CAPABILITIES}" "${CUDA_ARCHS}")
        if(NOT _tirx_archs)
            message(FATAL_ERROR "TIRx kernel: ${TIRX_NAME}, empty set of capabilities after intersection (kernel: ${TIRX_CUDA_CAPABILITIES}, supported: ${CUDA_ARCHS})")
        endif()
    else()
        set(_tirx_archs "${CUDA_KERNEL_ARCHS}")
    endif()
    list(SORT _tirx_archs COMPARE NATURAL)
    list(GET _tirx_archs 0 _tirx_arch)
    # Capabilities look like "8.9", "9.0a", or "9.0+PTX"; codegen wants "sm_89".
    string(REPLACE "+PTX" "" _tirx_arch "${_tirx_arch}")
    string(REPLACE "." "" _tirx_arch "${_tirx_arch}")
    if(NOT _tirx_arch MATCHES "^[0-9]+[a-z]?$")
        message(FATAL_ERROR "TIRx kernel: ${TIRX_NAME}, cannot derive codegen arch from `${_tirx_archs}`")
    endif()

    set(_tirx_codegen ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tirx_codegen.py)
    set(_tirx_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/tirx_generated)

    set(_tirx_gen_srcs)
    foreach(_tirx_src ${TIRX_SRC})
        get_filename_component(_tirx_stem ${_tirx_src} NAME_WE)
        set(_tirx_out ${_tirx_gen_dir}/${TIRX_NAME}_${_tirx_stem}.cu)
        add_custom_command(
            OUTPUT ${_tirx_out}
            COMMAND ${Python_EXECUTABLE} ${_tirx_codegen}
                --input ${CMAKE_CURRENT_SOURCE_DIR}/${_tirx_src}
                --output ${_tirx_out}
                --arch sm_${_tirx_arch}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_tirx_src} ${_tirx_codegen}
            COMMENT "TIRx codegen: ${_tirx_src} (sm_${_tirx_arch})"
            VERBATIM)
        list(APPEND _tirx_gen_srcs ${_tirx_out})
    endforeach()

    set_source_files_properties(${_tirx_gen_srcs} PROPERTIES LANGUAGE CUDA GENERATED TRUE)

    set(${SRC_VAR} ${_tirx_gen_srcs} PARENT_SCOPE)
endfunction()
