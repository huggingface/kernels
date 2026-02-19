cmake_minimum_required(VERSION 4.1)

enable_language(CUDA)
# Run `tvm-ffi-config --cmakedir` to set `tvm_ffi_ROOT`
find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tvm_ffi_ROOT)
find_package(tvm_ffi CONFIG REQUIRED)

# Link CUDA target to `tvm_ffi::header` and `tvm_ffi::shared`
add_library(relu SHARED relu_cuda/relu.cu)
tvm_ffi_configure_target(relu)
