# Avoid 'lib' prefix for the extension.
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# rust kernels export the symbols; CMake just needs a stub source for the shared library.
if(NOT SRC AND RUST_KERNEL_LIBS)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_ops_stub.cpp "\n")
  list(APPEND SRC ${CMAKE_CURRENT_BINARY_DIR}/_ops_stub.cpp)
endif()

add_library(${OPS_NAME} SHARED ${SRC})
target_compile_definitions(${OPS_NAME} PRIVATE
  "-DTVM_FFI_EXTENSION_NAME=${OPS_NAME}")
tvm_ffi_configure_target(${OPS_NAME})

if(RUST_KERNEL_LIBS)
  add_dependencies(${OPS_NAME} ${RUST_KERNEL_TARGETS})
  find_package(Threads REQUIRED)
  target_link_libraries(${OPS_NAME} PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,${RUST_KERNEL_LIBS}>"
    Threads::Threads
    ${CMAKE_DL_LIBS})

  if(GPU_LANG STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${OPS_NAME} PRIVATE CUDA::cuda_driver)
  endif()
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
