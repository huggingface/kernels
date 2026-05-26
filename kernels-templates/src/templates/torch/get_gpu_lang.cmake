#
# Get the GPU language from Torch.
#
function(get_gpu_lang OUT)
    run_python_script(PYTHON_OUT
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_gpu_lang.py"
        "Cannot detect GPU language")
    set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()
