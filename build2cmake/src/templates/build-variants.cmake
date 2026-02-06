# Generate a standardized build variant name following the pattern:
# torch<VERSION>-[cxx11-]<COMPUTE>-<ARCH>-<OS>
#
# Arguments:
#   OUT_BUILD_NAME - Output variable name
#   TORCH_VERSION - PyTorch version (e.g., "2.7.1")
#   COMPUTE_FRAMEWORK - One of: cuda, rocm, metal, xpu, cpu
#   COMPUTE_VERSION - Version of compute framework (e.g., "12.4" for CUDA, "6.0" for ROCm)
#                     Optional for CPU-only builds (pass empty string or omit)
# Example output: torch271-cxx11-cu124-x86_64-linux (Linux)
#                 torch271-cu124-x86_64-windows (Windows)
#                 torch271-metal-aarch64-darwin (macOS)
#
function(generate_build_name OUT_BUILD_NAME TORCH_VERSION COMPUTE_FRAMEWORK COMPUTE_VERSION)
    # Flatten version by removing dots and padding to 2 components
    string(REPLACE "." ";" VERSION_LIST "${TORCH_VERSION}")
    list(LENGTH VERSION_LIST VERSION_COMPONENTS)

    # Pad to at least 2 components
    if(VERSION_COMPONENTS LESS 2)
        list(APPEND VERSION_LIST "0")
    endif()

    # Take first 2 components and join without dots
    list(GET VERSION_LIST 0 MAJOR)
    list(GET VERSION_LIST 1 MINOR)
    set(FLATTENED_TORCH "${MAJOR}${MINOR}")

    # Generate compute string
    if(COMPUTE_FRAMEWORK STREQUAL "cuda")
        # Flatten CUDA version (e.g., "12.4" -> "124")
        string(REPLACE "." ";" COMPUTE_VERSION_LIST "${COMPUTE_VERSION}")
        list(LENGTH COMPUTE_VERSION_LIST COMPUTE_COMPONENTS)
        if(COMPUTE_COMPONENTS GREATER_EQUAL 2)
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            list(GET COMPUTE_VERSION_LIST 1 COMPUTE_MINOR)
            set(COMPUTE_STRING "cu${COMPUTE_MAJOR}${COMPUTE_MINOR}")
        else()
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            set(COMPUTE_STRING "cu${COMPUTE_MAJOR}0")
        endif()
    elseif(COMPUTE_FRAMEWORK STREQUAL "rocm")
        # Flatten ROCm version (e.g., "6.0" -> "60")
        string(REPLACE "." ";" COMPUTE_VERSION_LIST "${COMPUTE_VERSION}")
        list(LENGTH COMPUTE_VERSION_LIST COMPUTE_COMPONENTS)
        if(COMPUTE_COMPONENTS GREATER_EQUAL 2)
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            list(GET COMPUTE_VERSION_LIST 1 COMPUTE_MINOR)
            set(COMPUTE_STRING "rocm${COMPUTE_MAJOR}${COMPUTE_MINOR}")
        else()
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            set(COMPUTE_STRING "rocm${COMPUTE_MAJOR}0")
        endif()
    elseif(COMPUTE_FRAMEWORK STREQUAL "xpu")
        # Flatten XPU version (e.g., "2025.2" -> "202552")
        string(REPLACE "." ";" COMPUTE_VERSION_LIST "${COMPUTE_VERSION}")
        list(LENGTH COMPUTE_VERSION_LIST COMPUTE_COMPONENTS)
        if(COMPUTE_COMPONENTS GREATER_EQUAL 2)
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            list(GET COMPUTE_VERSION_LIST 1 COMPUTE_MINOR)
            set(COMPUTE_STRING "xpu${COMPUTE_MAJOR}${COMPUTE_MINOR}")
        else()
            list(GET COMPUTE_VERSION_LIST 0 COMPUTE_MAJOR)
            set(COMPUTE_STRING "xpu${COMPUTE_MAJOR}0")
        endif()
    elseif(COMPUTE_FRAMEWORK STREQUAL "metal")
        set(COMPUTE_STRING "metal")
    elseif(COMPUTE_FRAMEWORK STREQUAL "cpu")
        set(COMPUTE_STRING "cpu")
    else()
        message(FATAL_ERROR "Unknown compute framework: ${COMPUTE_FRAMEWORK}")
    endif()

    # Detect from target system (CMAKE_SYSTEM_* variables refer to target, not host)
    # Normalize architecture name
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        set(CPU_ARCH "x86_64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(CPU_ARCH "aarch64")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Normalize OS name
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(OS_NAME "windows")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(OS_NAME "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(OS_NAME "darwin")
    else()
        message(WARNING "Unknown OS ${CMAKE_SYSTEM_NAME}, using as-is")
        string(TOLOWER "${CMAKE_SYSTEM_NAME}" OS_NAME)
    endif()

    set(ARCH_OS_STRING "${CPU_ARCH}-${OS_NAME}")

    # Assemble the final build name
    # For Linux, include cxx11 ABI indicator for compatibility
    if(ARCH_OS_STRING MATCHES "-linux$")
        set(BUILD_NAME "torch${FLATTENED_TORCH}-cxx11-${COMPUTE_STRING}-${ARCH_OS_STRING}")
    else()
        set(BUILD_NAME "torch${FLATTENED_TORCH}-${COMPUTE_STRING}-${ARCH_OS_STRING}")
    endif()

    set(${OUT_BUILD_NAME} "${BUILD_NAME}" PARENT_SCOPE)
    message(STATUS "Generated build name: ${BUILD_NAME}")
endfunction()

#
# Create a custom install target for the huggingface/kernels library layout.
# This installs the extension into a directory structure suitable for kernel hub discovery:
#   <PREFIX>/<BUILD_VARIANT_NAME>
#
# Arguments:
#   TARGET_NAME - Name of the target to create the install rule for
#   PACKAGE_NAME - Python package name (e.g., "activation")
#   BUILD_VARIANT_NAME - Build variant name (e.g., "torch271-cxx11-cu124-x86_64-linux")
#   INSTALL_PREFIX - Base installation directory (defaults to CMAKE_INSTALL_PREFIX)
#
function(add_kernels_install_target TARGET_NAME PACKAGE_NAME BUILD_VARIANT_NAME)
    set(oneValueArgs INSTALL_PREFIX)
    set(multiValueArgs DATA_EXTENSIONS)
    cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_INSTALL_PREFIX)
        set(ARG_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()

    if (${GPU_LANG} STREQUAL "CPU")
        set(_BACKEND "cpu")
    elseif (${GPU_LANG} STREQUAL "CUDA")
        set(_BACKEND "cuda")
    elseif (${GPU_LANG} STREQUAL "HIP")
        set(_BACKEND "rocm")
    elseif (${GPU_LANG} STREQUAL "METAL")
        set(_BACKEND "metal")
    elseif (${GPU_LANG} STREQUAL "SYCL")
        set(_BACKEND "xpu")
    else()
        message(FATAL_ERROR "Unsupported GPU_LANG: ${GPU_LANG}")
    endif()

    # Set the installation directory
    set(KERNEL_INSTALL_DIR "${ARG_INSTALL_PREFIX}/${BUILD_VARIANT_NAME}")

    message(STATUS "Using PACKAGE_NAME: ${PACKAGE_NAME}")

    # Install the compiled extension using CMake's install() command
    # This will be triggered by the standard INSTALL target
    install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION "${KERNEL_INSTALL_DIR}"
        RUNTIME DESTINATION "${KERNEL_INSTALL_DIR}"
        COMPONENT ${TARGET_NAME})

    # Glob Python files to install recursively.
    file(GLOB_RECURSE PYTHON_FILES RELATIVE "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}" "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/*.py")
    foreach(python_file IN LISTS PYTHON_FILES)
        get_filename_component(python_file_dir "${python_file}" DIRECTORY)
        install(FILES "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/${python_file}"
          DESTINATION "${KERNEL_INSTALL_DIR}/${python_file_dir}"
          COMPONENT ${TARGET_NAME})
    endforeach()

    install(FILES ${CMAKE_SOURCE_DIR}/metadata-${_BACKEND}.json
        DESTINATION "${KERNEL_INSTALL_DIR}"
        RENAME "metadata.json"
        COMPONENT ${TARGET_NAME})

    # Compatibility with older kernels and direct Python imports.
    install(FILES ${CMAKE_SOURCE_DIR}/compat.py
      DESTINATION "${KERNEL_INSTALL_DIR}/${PACKAGE_NAME}"
        RENAME "__init__.py"
        COMPONENT ${TARGET_NAME})

    # Install data files with specified extensions
    foreach(ext IN LISTS ARG_DATA_EXTENSIONS)
        file(GLOB_RECURSE DATA_FILES RELATIVE "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}" "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/*.${ext}")
        foreach(data_file IN LISTS DATA_FILES)
            get_filename_component(data_file_dir "${data_file}" DIRECTORY)
            install(FILES "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/${data_file}"
                DESTINATION "${KERNEL_INSTALL_DIR}/${data_file_dir}"
                COMPONENT ${TARGET_NAME})
        endforeach()
    endforeach()

    message(STATUS "Added install rules for ${TARGET_NAME} -> ${BUILD_VARIANT_NAME}")
endfunction()

#
# Add install rules for local development with huggingface/kernels.
# This installs the extension into the layout expected by get_local_kernel():
#   ${CMAKE_SOURCE_DIR}/build/<BUILD_VARIANT_NAME>/
#
# This allows developers to use get_local_kernel() from the kernels library to load
# locally built kernels without needing to publish to the hub.
#
# This uses the standard CMake install() command, so it works with the default
# "install" target that is always available.
#
# Arguments:
#   TARGET_NAME - Name of the target to create the install rule for
#   PACKAGE_NAME - Python package name (e.g., "activation")
#   BUILD_VARIANT_NAME - Build variant name (e.g., "torch271-cxx11-cu124-x86_64-linux")
#
function(add_local_install_target TARGET_NAME PACKAGE_NAME BUILD_VARIANT_NAME)
    set(multiValueArgs DATA_EXTENSIONS)
    cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})

    # Define your local, folder based, installation directory
    set(LOCAL_INSTALL_DIR "${CMAKE_SOURCE_DIR}/build/${BUILD_VARIANT_NAME}")
    # Variant directory is where metadata.json should go (for kernels upload discovery)
    set(VARIANT_DIR "${CMAKE_SOURCE_DIR}/build/${BUILD_VARIANT_NAME}")

    # Glob Python files to install recursively.
    file(GLOB_RECURSE PYTHON_FILES RELATIVE "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}" "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/*.py")

    # Create a custom target for local installation
    add_custom_target(local_install
            COMMENT "Installing files to local directory..."
    )

    if (${GPU_LANG} STREQUAL "CPU")
        set(_BACKEND "cpu")
    elseif (${GPU_LANG} STREQUAL "CUDA")
        set(_BACKEND "cuda")
    elseif (${GPU_LANG} STREQUAL "HIP")
        set(_BACKEND "rocm")
    elseif (${GPU_LANG} STREQUAL "METAL")
        set(_BACKEND "metal")
    elseif (${GPU_LANG} STREQUAL "SYCL")
        set(_BACKEND "xpu")
    else()
        message(FATAL_ERROR "Unsupported GPU_LANG: ${GPU_LANG}")
    endif()

    # Add custom commands to copy files
    add_custom_command(TARGET local_install POST_BUILD
            # Copy the shared library
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${TARGET_NAME}>
            ${LOCAL_INSTALL_DIR}/

            # Copy metadata.json if it exists
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/metadata-${_BACKEND}.json
            ${VARIANT_DIR}/metadata.json

            # Compatibility with older kernels and direct Python imports.
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/compat.py
            ${VARIANT_DIR}/${PACKAGE_NAME}/__init__.py

            COMMENT "Copying shared library and Python files to ${LOCAL_INSTALL_DIR}"
            COMMAND_EXPAND_LISTS
    )

    # Copy each Python file preserving directory structure
    foreach(python_file IN LISTS PYTHON_FILES)
        get_filename_component(python_file_dir "${python_file}" DIRECTORY)
        add_custom_command(TARGET local_install POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E make_directory
              ${LOCAL_INSTALL_DIR}/${python_file_dir}
              COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/${python_file}
              ${LOCAL_INSTALL_DIR}/${python_file_dir}/
              COMMENT "Copying ${python_file} to ${LOCAL_INSTALL_DIR}/${python_file_dir}"
      )
    endforeach()

    # Copy data files with specified extensions
    foreach(ext IN LISTS ARG_DATA_EXTENSIONS)
        file(GLOB_RECURSE DATA_FILES RELATIVE "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}" "${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/*.${ext}")
        foreach(data_file IN LISTS DATA_FILES)
            get_filename_component(data_file_dir "${data_file}" DIRECTORY)
            add_custom_command(TARGET local_install POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory
                ${LOCAL_INSTALL_DIR}/${data_file_dir}
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_SOURCE_DIR}/torch-ext/${PACKAGE_NAME}/${data_file}
                ${LOCAL_INSTALL_DIR}/${data_file_dir}/
                COMMENT "Copying ${data_file} to ${LOCAL_INSTALL_DIR}/${data_file_dir}"
            )
        endforeach()
    endforeach()

    # Create both directories: variant dir for metadata.json, package dir for binaries
    file(MAKE_DIRECTORY ${VARIANT_DIR})
    file(MAKE_DIRECTORY ${LOCAL_INSTALL_DIR})
    message(STATUS "Added install rules for ${TARGET_NAME} -> build/${BUILD_VARIANT_NAME}")
endfunction()
