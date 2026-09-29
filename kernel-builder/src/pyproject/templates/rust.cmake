function(rust_kernel_component LIBS_VAR TARGETS_VAR)
    cmake_parse_arguments(KERNEL "" "NAME;MANIFEST_PATH" "" ${ARGN})

    if(NOT KERNEL_NAME OR NOT KERNEL_MANIFEST_PATH)
        message(FATAL_ERROR "rust_kernel_component: NAME and MANIFEST_PATH are required")
    endif()

    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR "Kernel component `${KERNEL_NAME}` is written in Rust, "
            "but `cargo` was not found. Install a Rust toolchain or set CARGO_EXECUTABLE.")
    endif()

    string(REPLACE "-" "_" _LIB_NAME ${KERNEL_NAME})

    set(_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME})
    set(_STATICLIB ${_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})

    # tvm-ffi-sys's build script shells out to `tvm-ffi-config`, a console script
    # the apache-tvm-ffi wheel installs beside the interpreter.
    get_filename_component(_PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)

    add_custom_target(${KERNEL_NAME}_cargo_build ALL
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${_PYTHON_BIN_DIR}:$ENV{PATH}"
            ${CARGO_EXECUTABLE} rustc --release --locked --lib --crate-type staticlib
            --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_MANIFEST_PATH}
            --target-dir ${_CARGO_TARGET_DIR}
        BYPRODUCTS ${_STATICLIB}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust kernel ${KERNEL_NAME} with cargo"
        VERBATIM
    )

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
