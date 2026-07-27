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

function(rust_kernel_component LIBS_VAR TARGETS_VAR)
    set(oneValueArgs NAME MANIFEST_PATH LIB_NAME DEVICE_MANIFEST PTX_DIR)
    set(multiValueArgs FEATURES CUDA_CAPABILITIES)
    cmake_parse_arguments(KERNEL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT KERNEL_NAME OR NOT KERNEL_MANIFEST_PATH OR NOT KERNEL_LIB_NAME)
        message(FATAL_ERROR "rust_kernel_component requires NAME, MANIFEST_PATH, and LIB_NAME")
    endif()

    find_program(CARGO_EXECUTABLE cargo REQUIRED)

    # Export CUDA archs for Rust build scripts.
    set(_KERNEL_ARCHS "")
    if(GPU_LANG STREQUAL "CUDA")
        if(KERNEL_CUDA_CAPABILITIES)
            cuda_archs_loose_intersection(_KERNEL_ARCHS "${KERNEL_CUDA_CAPABILITIES}" "${CUDA_ARCHS}")
            if(NOT _KERNEL_ARCHS)
                message(FATAL_ERROR "Rust kernel: ${KERNEL_NAME}, empty set of capabilities after intersection (kernel: ${KERNEL_CUDA_CAPABILITIES}, supported: ${CUDA_ARCHS})")
            endif()
        else()
            set(_KERNEL_ARCHS "${CUDA_KERNEL_ARCHS}")
        endif()
        message(STATUS "Rust kernel: ${KERNEL_NAME}, capabilities: ${_KERNEL_ARCHS}")

        accumulate_gpu_archs(_ALL_GPU_ARCHS "${ALL_GPU_ARCHS}" "${_KERNEL_ARCHS}")
        set(ALL_GPU_ARCHS ${_ALL_GPU_ARCHS} PARENT_SCOPE)
    endif()

    # Cargo writes the staticlib into its target directory.
    set(_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME})
    set(_STATICLIB ${_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}${KERNEL_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_RUST_KERNEL_DEPENDS)

    if(KERNEL_DEVICE_MANIFEST)
        # Build cuda-oxide device code before the host crate embeds its PTX.
        if(NOT DEFINED ENV{CUDA_OXIDE_BACKEND} OR "$ENV{CUDA_OXIDE_BACKEND}" STREQUAL "")
            message(FATAL_ERROR "rust_kernel_component: DEVICE_MANIFEST requires CUDA_OXIDE_BACKEND")
        endif()
        if(NOT KERNEL_PTX_DIR)
            set(KERNEL_PTX_DIR kernels-ptx)
        endif()

        get_filename_component(_DEVICE_MANIFEST ${KERNEL_DEVICE_MANIFEST} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        get_filename_component(_PTX_DIR ${KERNEL_PTX_DIR} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
        set(_DEVICE_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME}-device)
        set(_DEVICE_ENV)

        if(_KERNEL_ARCHS)
            list(SORT _KERNEL_ARCHS COMPARE NATURAL)
            # PTX JITs forward, so target the lowest supported arch.
            list(GET _KERNEL_ARCHS 0 _OXIDE_ARCH)
            string(REPLACE "+PTX" "" _OXIDE_ARCH "${_OXIDE_ARCH}")
            string(REPLACE "." "" _OXIDE_ARCH "${_OXIDE_ARCH}")
            list(APPEND _DEVICE_ENV "CUDA_OXIDE_TARGET=sm_${_OXIDE_ARCH}")
        endif()

        add_custom_target(${KERNEL_NAME}_oxide_device_build ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_PTX_DIR}
            COMMAND ${CMAKE_COMMAND} -E env
                "CUDA_OXIDE_PTX_DIR=${_PTX_DIR}"
                "RUSTFLAGS=-Zcodegen-backend=$ENV{CUDA_OXIDE_BACKEND} -Copt-level=3 -Cdebug-assertions=off -Zmir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"
                ${_DEVICE_ENV}
                ${CARGO_EXECUTABLE} build --release --locked
                    --manifest-path ${_DEVICE_MANIFEST}
                    --target-dir ${_DEVICE_TARGET_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building Rust CUDA device kernel ${KERNEL_NAME}"
            VERBATIM
        )
        list(APPEND _RUST_KERNEL_DEPENDS ${KERNEL_NAME}_oxide_device_build)
    endif()

    set(_CARGO_ARGS rustc --release --locked --lib --crate-type staticlib
        --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_MANIFEST_PATH}
        --target-dir ${_CARGO_TARGET_DIR})
    if(KERNEL_FEATURES)
        list(JOIN KERNEL_FEATURES "," _KERNEL_FEATURES)
        list(APPEND _CARGO_ARGS --features ${_KERNEL_FEATURES})
    endif()

    get_filename_component(_PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)

    add_custom_target(${KERNEL_NAME}_cargo_build ALL
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=${_PYTHON_BIN_DIR}:$ENV{PATH}"
            "KERNEL_BUILDER_GPU_LANG=${GPU_LANG}"
            "KERNEL_BUILDER_CUDA_ARCHS=${_KERNEL_ARCHS}"
            ${CARGO_EXECUTABLE} ${_CARGO_ARGS}
        BYPRODUCTS ${_STATICLIB}
        DEPENDS ${_RUST_KERNEL_DEPENDS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust kernel ${KERNEL_NAME} with cargo"
        VERBATIM
    )

    # Register the cargo artifact as a CMake library target.
    add_library(${KERNEL_NAME}_rust STATIC IMPORTED GLOBAL)
    set_target_properties(${KERNEL_NAME}_rust PROPERTIES
        IMPORTED_LOCATION ${_STATICLIB})

    # Return the library and build target to the extension scope.
    set(${LIBS_VAR} ${${LIBS_VAR}} ${KERNEL_NAME}_rust PARENT_SCOPE)
    set(${TARGETS_VAR} ${${TARGETS_VAR}} ${KERNEL_NAME}_cargo_build PARENT_SCOPE)
endfunction()
