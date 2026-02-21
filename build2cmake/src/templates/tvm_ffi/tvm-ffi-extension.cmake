# Avoid 'lib' prefix for the extension.
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${OPS_NAME} SHARED ${SRC})
target_compile_definitions(${OPS_NAME} PRIVATE
  "-DTVM_FFI_EXTENSION_NAME=${OPS_NAME}")
tvm_ffi_configure_target(${OPS_NAME})
