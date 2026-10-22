# Functions run with the caller's CMAKE_CURRENT_LIST_DIR, so remember where the
# templates beside this file are.
set(_RUST_CMAKE_DIR ${CMAKE_CURRENT_LIST_DIR})

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

        # The crate is built with the cuda-oxide codegen backend, which compiles
        # host code as usual and writes the device code as PTX.
        if(NOT CUDA_OXIDE_BACKEND)
            message(FATAL_ERROR "rust_kernel_component: Rust CUDA kernels require CUDA_OXIDE_BACKEND")
        endif()

        # The fatbin is linked in under fixed symbol names.
        get_property(_FATBIN_OWNER GLOBAL PROPERTY RUST_CUDA_FATBIN_OWNER)
        if(_FATBIN_OWNER)
            message(FATAL_ERROR "rust_kernel_component: ${KERNEL_NAME} and ${_FATBIN_OWNER} "
                "are both Rust CUDA components; only one is supported per extension")
        endif()
        set_property(GLOBAL PROPERTY RUST_CUDA_FATBIN_OWNER ${KERNEL_NAME})

        get_filename_component(_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
        find_program(CUDA_PTXAS_EXECUTABLE ptxas HINTS ${_CUDA_BIN_DIR} REQUIRED)
        find_program(CUDA_FATBINARY_EXECUTABLE fatbinary HINTS ${_CUDA_BIN_DIR} REQUIRED)
        if(NOT CMAKE_OBJCOPY)
            message(FATAL_ERROR "rust_kernel_component: Rust CUDA kernels require objcopy")
        endif()

        list(SORT _KERNEL_ARCHS COMPARE NATURAL)
        # ptxas compiles PTX for newer archs too, so target the lowest one.
        list(GET _KERNEL_ARCHS 0 _OXIDE_ARCH)
        string(REPLACE "+PTX" "" _OXIDE_ARCH "${_OXIDE_ARCH}")
        string(REPLACE "." "" _OXIDE_ARCH "${_OXIDE_ARCH}")
        # Flags match what `cargo oxide build` passes to the codegen backend.
        list(APPEND _CARGO_ENV
            "CUDA_OXIDE_TARGET=sm_${_OXIDE_ARCH}"
            "CUDA_OXIDE_PTX_DIR=${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME}/ptx"
            "RUSTFLAGS=-Zcodegen-backend=${CUDA_OXIDE_BACKEND} -Copt-level=3 -Cdebug-assertions=off -Zmir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0")
    endif()

    string(REPLACE "-" "_" _LIB_NAME ${KERNEL_NAME})

    set(_CARGO_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/${KERNEL_NAME})
    set(_STATICLIB ${_CARGO_TARGET_DIR}/release/${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(_LINKED_STATICLIB ${_STATICLIB})
    set(_BYPRODUCTS ${_STATICLIB})

    if(GPU_LANG STREQUAL "CUDA")
        set(_PTX ${_CARGO_TARGET_DIR}/ptx/${_LIB_NAME}.ptx)
        list(APPEND _BYPRODUCTS ${_PTX})
        # The backend also embeds the PTX in the crate's `.oxart` sections. The
        # extension loads the fatbin instead, so drop the PTX from what we link.
        set(_LINKED_STATICLIB ${_CARGO_TARGET_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${_LIB_NAME}_sass${CMAKE_STATIC_LIBRARY_SUFFIX})
        list(APPEND _BYPRODUCTS ${_LINKED_STATICLIB})
        # Keeping the date lets a no-op cargo build skip the relink.
        set(_STRIP_PTX COMMAND ${CMAKE_OBJCOPY} --preserve-dates --remove-section=.oxart
            ${_STATICLIB} ${_LINKED_STATICLIB})
    endif()

    # tvm-ffi-sys's build script shells out to `tvm-ffi-config`, a console script
    # the apache-tvm-ffi wheel installs beside the interpreter.
    get_filename_component(_PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)

    add_custom_target(${KERNEL_NAME}_cargo_build ALL
        COMMAND ${CMAKE_COMMAND} -E env "PATH=${_PYTHON_BIN_DIR}:$ENV{PATH}"
            ${_CARGO_ENV}
            ${CARGO_EXECUTABLE} rustc --release --locked --lib --crate-type staticlib
            --manifest-path ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_MANIFEST_PATH}
            --target-dir ${_CARGO_TARGET_DIR}
        ${_STRIP_PTX}
        BYPRODUCTS ${_BYPRODUCTS}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building Rust kernel ${KERNEL_NAME} with cargo"
        VERBATIM
    )

    add_library(${KERNEL_NAME}_rust STATIC IMPORTED GLOBAL)
    set_target_properties(${KERNEL_NAME}_rust PROPERTIES IMPORTED_LOCATION ${_LINKED_STATICLIB})

    if(GPU_LANG STREQUAL "CUDA")
        _rust_cuda_fatbin(${KERNEL_NAME} ${_PTX} ${_OXIDE_ARCH} "${_KERNEL_ARCHS}")
    endif()

    set(${LIBS_VAR} ${${LIBS_VAR}} ${KERNEL_NAME}_rust PARENT_SCOPE)
    set(${TARGETS_VAR} ${${TARGETS_VAR}} ${KERNEL_NAME}_cargo_build PARENT_SCOPE)
endfunction()

# Compiles the crate's PTX to SASS for every capability, so nothing is
# JIT-compiled on the user's machine. `target_link_rust_kernels` links the
# fatbin between `KERNEL_BUILDER_CUDA_FATBIN` and `KERNEL_BUILDER_CUDA_FATBIN_END`.
function(_rust_cuda_fatbin NAME PTX PTX_ARCH ARCHS)
    get_filename_component(_DIR ${PTX} DIRECTORY)
    set(_CUBINS)
    set(_IMAGES)
    foreach(_ARCH IN LISTS ARCHS)
        string(REPLACE "+PTX" "" _SM "${_ARCH}")
        string(REPLACE "." "" _SM "${_SM}")
        set(_CUBIN ${_DIR}/${NAME}.sm_${_SM}.cubin)
        add_custom_command(OUTPUT ${_CUBIN}
            COMMAND ${CUDA_PTXAS_EXECUTABLE} -arch=sm_${_SM} ${PTX} -o ${_CUBIN}
            DEPENDS ${PTX}
            COMMENT "Compiling Rust kernel ${NAME} for sm_${_SM}"
            VERBATIM)
        list(APPEND _CUBINS ${_CUBIN})
        list(APPEND _IMAGES "--image3=kind=elf,sm=${_SM},file=${_CUBIN}")
        if(_ARCH MATCHES "\\+PTX$")
            set(_KEEP_PTX ON)
        endif()
    endforeach()
    if(_KEEP_PTX)
        # The PTX targets the lowest arch, so it JITs on any newer GPU.
        list(APPEND _IMAGES "--image3=kind=ptx,sm=${PTX_ARCH},file=${PTX}")
    endif()

    set(_FATBIN ${_DIR}/${NAME}.fatbin)
    add_custom_command(OUTPUT ${_FATBIN}
        COMMAND ${CUDA_FATBINARY_EXECUTABLE} --create=${_FATBIN} ${_IMAGES}
        DEPENDS ${_CUBINS} ${PTX}
        COMMENT "Creating fatbin for Rust kernel ${NAME}"
        VERBATIM)

    set(_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}_fatbin.cpp)
    set(FATBIN ${_FATBIN})
    configure_file(${_RUST_CMAKE_DIR}/rust-fatbin.cpp.in ${_SOURCE} @ONLY)
    set_source_files_properties(${_SOURCE} PROPERTIES OBJECT_DEPENDS ${_FATBIN})
    set_property(GLOBAL PROPERTY RUST_CUDA_FATBIN_SOURCE ${_SOURCE})
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
    if(UNIX AND NOT APPLE)
        # The codegen backend's embedded PTX objects carry no `.note.GNU-stack`,
        # which would otherwise make the linker mark the stack executable.
        target_link_options(${TARGET} PRIVATE "LINKER:-z,noexecstack")
        get_property(_FATBIN_SOURCE GLOBAL PROPERTY RUST_CUDA_FATBIN_SOURCE)
        if(_FATBIN_SOURCE)
            target_sources(${TARGET} PRIVATE ${_FATBIN_SOURCE})
            # A crate that still calls the generated `load` references the
            # embedded PTX we removed. Fail the link instead of the import.
            target_link_options(${TARGET} PRIVATE "LINKER:--no-undefined")
        endif()
    endif()
endfunction()
