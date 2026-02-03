#
# Get the GPU language from Torch.
#
function(get_gpu_lang OUT)
    execute_process(
    COMMAND
    "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_gpu_lang.py"
    OUTPUT_VARIABLE PYTHON_OUT
    RESULT_VARIABLE PYTHON_ERROR_CODE
    ERROR_VARIABLE PYTHON_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(NOT PYTHON_ERROR_CODE EQUAL 0)
        message(FATAL_ERROR "Cannot detect GPU language: ${PYTHON_STDERR}")
    endif()
    set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()
