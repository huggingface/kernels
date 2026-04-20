# Avoid 'lib' prefix for the extension.
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${OPS_NAME} SHARED ${SRC})
target_compile_definitions(${OPS_NAME} PRIVATE
  "-DTVM_FFI_EXTENSION_NAME=${OPS_NAME}")
tvm_ffi_configure_target(${OPS_NAME})

if(NOT (MSVC OR GPU_LANG STREQUAL "SYCL"))
   target_link_options(${OPS_NAME} PRIVATE -static-libstdc++)
endif()

if(GPU_LANG STREQUAL "SYCL")
    target_link_options(${OPS_NAME} PRIVATE ${sycl_link_flags})
    target_link_libraries(${OPS_NAME} PRIVATE dnnl)
endif()

# Compile Metal shaders if any were found
if(GPU_LANG STREQUAL "METAL")
    if(ALL_METAL_SOURCES)
        compile_metal_shaders(${OPS_NAME} "${ALL_METAL_SOURCES}" "${METAL_INCLUDE_DIRS}")
    endif()
endif()

install(TARGETS ${OPS_NAME} LIBRARY DESTINATION ${OPS_NAME} COMPONENT ${OPS_NAME})
# Add kernels_install target for huggingface/kernels library layout
add_kernels_install_target(${OPS_NAME} "{{ python_name }}" "${BUILD_VARIANT_NAME}"
    DATA_EXTENSIONS "{{ data_extensions | join(';') }}"
    GPU_ARCHS "${ALL_GPU_ARCHS}")

# Add local_install target for local development with get_local_kernel()
add_local_install_target(${OPS_NAME} "{{ python_name }}" "${BUILD_VARIANT_NAME}"
    DATA_EXTENSIONS "{{ data_extensions | join(';') }}"
    GPU_ARCHS "${ALL_GPU_ARCHS}")
