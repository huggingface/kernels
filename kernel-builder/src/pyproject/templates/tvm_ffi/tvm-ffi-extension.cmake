# Avoid 'lib' prefix for the extension.
set(CMAKE_SHARED_LIBRARY_PREFIX "")

rust_extension_sources(SRC)

# Add hipify preprocessing step when building with HIP/ROCm.
if(GPU_LANG STREQUAL "HIP")
  hipify_sources_target(SRC ${OPS_NAME} "${SRC}")
endif()

add_library(${OPS_NAME} SHARED ${SRC})

if(GPU_LANG STREQUAL "HIP")
  # Make this target dependent on the hipify preprocessor step.
  add_dependencies(${OPS_NAME} hipify${OPS_NAME})

  # Clear target architectures, we are passing arch flags per source file.
  set_property(TARGET ${OPS_NAME} PROPERTY HIP_ARCHITECTURES off)
endif()
target_compile_definitions(${OPS_NAME} PRIVATE
  "-DTVM_FFI_EXTENSION_NAME=${OPS_NAME}")
tvm_ffi_configure_target(${OPS_NAME})

# Avoid that definitions of template static data members and static local
# variables in inline functions collide. Leads to subtle bugs between kernels
# that happen to have clashes.
check_cxx_compiler_flag("-fno-gnu-unique" CXX_HAS_NO_GNU_UNIQUE)
if(CXX_HAS_NO_GNU_UNIQUE)
  target_compile_options(${OPS_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fno-gnu-unique>)
  target_compile_options(${OPS_NAME} PRIVATE $<$<COMPILE_LANGUAGE:GPU_LANGUAGE>:-fno-gnu-unique>)
endif()

target_link_rust_kernels(${OPS_NAME})

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
