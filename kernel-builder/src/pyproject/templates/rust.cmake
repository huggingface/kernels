# cuda-oxide interop: device crates are declared by the host crate in
# `[[package.metadata.cuda-oxide.device-crates]]`, as `cargo oxide` reads them.
function(_cuda_oxide_device_manifests OUT_VAR MANIFEST)
    execute_process(
        COMMAND ${CARGO_EXECUTABLE} metadata --format-version 1 --no-deps --offline
            --manifest-path ${MANIFEST}
        OUTPUT_VARIABLE _JSON
        ERROR_VARIABLE _ERROR
        RESULT_VARIABLE _RESULT)
    if(NOT _RESULT EQUAL 0)
        message(FATAL_ERROR "cargo metadata failed for ${MANIFEST}: ${_ERROR}")
    endif()

    file(REAL_PATH ${MANIFEST} _MANIFEST)
    set(_MANIFESTS)
    string(JSON _N_PACKAGES LENGTH "${_JSON}" packages)
    math(EXPR _LAST_PACKAGE "${_N_PACKAGES} - 1")
    foreach(_I RANGE ${_LAST_PACKAGE})
        string(JSON _PACKAGE_MANIFEST GET "${_JSON}" packages ${_I} manifest_path)
        file(REAL_PATH ${_PACKAGE_MANIFEST} _PACKAGE_MANIFEST)
        if(NOT _PACKAGE_MANIFEST STREQUAL _MANIFEST)
            continue()
        endif()

        string(JSON _CRATES ERROR_VARIABLE _NO_CRATES
            GET "${_JSON}" packages ${_I} metadata cuda-oxide device-crates)
        if(_NO_CRATES)
            break()
        endif()
        get_filename_component(_PACKAGE_DIR ${_PACKAGE_MANIFEST} DIRECTORY)
        string(JSON _N_CRATES LENGTH "${_CRATES}")
        math(EXPR _LAST_CRATE "${_N_CRATES} - 1")
        foreach(_J RANGE ${_LAST_CRATE})
            string(JSON _CRATE_MANIFEST GET "${_CRATES}" ${_J} manifest-path)
            get_filename_component(_CRATE_MANIFEST ${_CRATE_MANIFEST} ABSOLUTE BASE_DIR ${_PACKAGE_DIR})
            list(APPEND _MANIFESTS ${_CRATE_MANIFEST})
        endforeach()
    endforeach()

    set(${OUT_VAR} ${_MANIFESTS} PARENT_SCOPE)
endfunction()

function(rust_kernel_component LIBS_VAR TARGETS_VAR)
    cmake_parse_arguments(KERNEL "" "NAME;MANIFEST_PATH" "CUDA_CAPABILITIES" ${ARGN})

    if(NOT KERNEL_NAME OR NOT KERNEL_MANIFEST_PATH)
        message(FATAL_ERROR "rust_kernel_component: NAME and MANIFEST_PATH are required")
    endif()

    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR "Kernel component `${KERNEL_NAME}` is written in Rust, "
            "but `cargo` was not found. Install a Rust toolchain or set CARGO_EXECUTABLE.")
    endif()

    if(GPU_LANG STREQUAL "CUDA")
        if(KERNEL_CUDA_CAPABILITIES)
            cuda_archs_intersection(_KERNEL_ARCHS "${KERNEL_CUDA_CAPABILITIES}" "${CUDA_ARCHS}")
            if(NOT _KERNEL_ARCHS)
                message(FATAL_ERROR "Rust kernel: ${KERNEL_NAME}, empty set of capabilities after intersection (kernel: ${KERNEL_CUDA_CAPABILITIES}, supported: ${CUDA_ARCHS})")
            endif()
        else()
            set(_KERNEL_ARCHS "${CUDA_KERNEL_ARCHS}")
        endif()
        message(STATUS "Rust kernel: ${KERNEL_NAME}, capabilities: ${_KERNEL_ARCHS}")

        accumulate_gpu_archs(_ALL_GPU_ARCHS "${ALL_GPU_ARCHS}" "${_KERNEL_ARCHS}")
        set(ALL_GPU_ARCHS ${_ALL_GPU_ARCHS} PARENT_SCOPE)

        _cuda_oxide_device_manifests(_DEVICE_MANIFESTS
            ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_MANIFEST_PATH})
    endif()

    set(_HOST_ENV)

    if(_DEVICE_MANIFESTS)
        # Build cuda-oxide device code before the host crate embeds its PTX.
        if(NOT CUDA_OXIDE_BACKEND)
            message(FATAL_ERROR "rust_kernel_component: cuda-oxide device crates require CUDA_OXIDE_BACKEND")
        endif()

        set(_DEVICE_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME}-device)
        set(_PTX_DIR ${_DEVICE_TARGET_DIR}/ptx)
        set(_DEVICE_ENV)

        if(_KERNEL_ARCHS)
            list(SORT _KERNEL_ARCHS COMPARE NATURAL)
            # PTX JITs forward, so target the lowest supported arch.
            list(GET _KERNEL_ARCHS 0 _OXIDE_ARCH)
            string(REPLACE "+PTX" "" _OXIDE_ARCH "${_OXIDE_ARCH}")
            string(REPLACE "." "" _OXIDE_ARCH "${_OXIDE_ARCH}")
            list(APPEND _DEVICE_ENV "CUDA_OXIDE_TARGET=sm_${_OXIDE_ARCH}")
        endif()

        set(_DEVICE_COMMANDS)
        foreach(_DEVICE_MANIFEST ${_DEVICE_MANIFESTS})
            # Flags match what `cargo oxide build` passes to the codegen backend.
            list(APPEND _DEVICE_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E env
                    "CUDA_OXIDE_PTX_DIR=${_PTX_DIR}"
                    "RUSTFLAGS=-Zcodegen-backend=${CUDA_OXIDE_BACKEND} -Copt-level=3 -Cdebug-assertions=off -Zmir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0"
                    ${_DEVICE_ENV}
                    ${CARGO_EXECUTABLE} build --release --locked
                        --manifest-path ${_DEVICE_MANIFEST}
                        --target-dir ${_DEVICE_TARGET_DIR})
        endforeach()

        add_custom_target(${KERNEL_NAME}_oxide_device_build ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_PTX_DIR}
            ${_DEVICE_COMMANDS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Building Rust CUDA device kernel ${KERNEL_NAME}"
            VERBATIM
        )
        list(APPEND _HOST_ENV "KERNEL_BUILDER_PTX_DIR=${_PTX_DIR}")
    endif()

    string(REPLACE "-" "_" _LIB_NAME ${KERNEL_NAME})

    set(_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME})
    set(_STATICLIB ${_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

    # tvm-ffi-sys's build script shells out to `tvm-ffi-config`, a console script
    # the apache-tvm-ffi wheel installs beside the interpreter.
    get_filename_component(_PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)

    add_custom_target(${KERNEL_NAME}_cargo_build ALL
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${_PYTHON_BIN_DIR}:$ENV{PATH}"
            ${_HOST_ENV}
            ${CARGO_EXECUTABLE} rustc --release --locked --lib --crate-type staticlib
            --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_MANIFEST_PATH}
            --target-dir ${_CARGO_TARGET_DIR}
        BYPRODUCTS ${_STATICLIB}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust kernel ${KERNEL_NAME} with cargo"
        VERBATIM
    )
    if(_DEVICE_MANIFESTS)
        add_dependencies(${KERNEL_NAME}_cargo_build ${KERNEL_NAME}_oxide_device_build)
    endif()

    add_library(${KERNEL_NAME}_rust STATIC IMPORTED GLOBAL)
    set_target_properties(${KERNEL_NAME}_rust PROPERTIES IMPORTED_LOCATION ${_STATICLIB})

    set(${LIBS_VAR} ${${LIBS_VAR}} ${KERNEL_NAME}_rust PARENT_SCOPE)
    set(${TARGETS_VAR} ${${TARGETS_VAR}} ${KERNEL_NAME}_cargo_build PARENT_SCOPE)
endfunction()

# `add_library(SHARED)` errors on an empty source list, and a Rust-only
# extension has none: the crate exports the tvm-ffi entry points itself.
function(rust_extension_sources SRC_VAR)
    if(${SRC_VAR})
        return()
    endif()
    if(NOT RUST_KERNEL_LIBS)
        message(FATAL_ERROR "No sources for the ${BACKEND} extension. Set "
            "`[tvm-ffi].src` or give this backend a kernel component.")
    endif()

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_ops_stub.cpp "\n")
    set(${SRC_VAR} ${${SRC_VAR}} ${CMAKE_CURRENT_BINARY_DIR}/_ops_stub.cpp PARENT_SCOPE)
endfunction()

# Whole-archive linking publishes the crate's bundled `std` at default
# visibility, so export only the names tvm-ffi resolves at load time.
function(_restrict_rust_exports TARGET)
    set(_EXPORTS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}-rust-exports)
    if(APPLE)
        file(WRITE ${_EXPORTS} "___tvm_ffi_*\n")
        set(_FLAG "-exported_symbols_list,${_EXPORTS}")
    elseif(UNIX)
        file(WRITE ${_EXPORTS} "{ global: __tvm_ffi_*; local: *; };\n")
        set(_FLAG "--version-script=${_EXPORTS}")
    else()
        message(FATAL_ERROR "Rust kernels are not supported on this platform "
            "(cannot restrict exported symbols to the tvm-ffi entry points)")
    endif()

    target_link_options(${TARGET} PRIVATE "LINKER:${_FLAG}")
    set_property(TARGET ${TARGET} APPEND PROPERTY LINK_DEPENDS ${_EXPORTS})
endfunction()

function(target_link_rust_kernels TARGET)
    if(NOT RUST_KERNEL_LIBS)
        return()
    endif()

    find_package(Threads REQUIRED)
    add_dependencies(${TARGET} ${RUST_KERNEL_TARGETS})
    target_link_libraries(${TARGET} PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,${RUST_KERNEL_LIBS}>"
        Threads::Threads
        ${CMAKE_DL_LIBS})
    _restrict_rust_exports(${TARGET})
endfunction()
